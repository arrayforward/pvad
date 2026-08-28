# -*- coding: utf-8 -*-
"""预计算混合物 fbank 和 enrollment 的 CAM++ embedding。

用法: python scripts/precompute_features.py --dir data/mixtures/val [--workers 8]
输出: <dir>/feats/<id>.npy  ([T,80] float32, per-utterance 均值归一化)
      <dir>/emb/<id>.npy    ([192] float32, L2 归一化)
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvad_common import ROOT, fbank, mean_normalize, load_labels  # noqa: E402

_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from pvad_common import CampplusEmbedder
        _EMBEDDER = CampplusEmbedder(intra_threads=1)
    return _EMBEDDER


def cmvn_ema_np(x, alpha=0.02):
    """因果 EMA 均值归一化 (流式部署一致)。"""
    import numpy as np
    out = np.empty_like(x)
    m = np.zeros(x.shape[1], dtype=np.float64)
    for t in range(len(x)):
        m = (1 - alpha) * m + alpha * x[t]
        out[t] = x[t] - m
    return out


def process_one(args):
    uid, mix_path, enroll_path, feat_out, emb_out, cmvn_mode = args
    try:
        if not Path(feat_out).exists():
            from pvad_common import read_wav
            pcm, sr = read_wav(ROOT / mix_path)
            feats = fbank(pcm)
            if cmvn_mode == "ema":
                feats = cmvn_ema_np(feats.astype(np.float64)).astype(np.float32)
            else:
                feats = mean_normalize(feats)
            np.save(feat_out, feats.astype(np.float32))
        if not Path(emb_out).exists():
            from pvad_common import read_wav
            pcm, sr = read_wav(ROOT / enroll_path)
            emb = _get_embedder().embed(pcm)
            np.save(emb_out, emb.astype(np.float32))
        return uid, None
    except Exception as e:  # noqa: BLE001
        return uid, str(e)


def process_tokens(args):
    """多帧 enrollment 表示: 1s 子段各过一个 CAM++ -> [N,192]。
    规则: 整 1s (16000 采样) 切分, 不足 1s 的尾段丢弃; enrollment >=3s 故 N>=3。"""
    uid, enroll_path, out_path = args
    try:
        if Path(out_path).exists():
            return uid, None
        from pvad_common import read_wav
        pcm, sr = read_wav(ROOT / enroll_path)
        emb = _get_embedder()
        n = len(pcm) // 16000
        toks = [emb.embed(pcm[i * 16000:(i + 1) * 16000]) for i in range(n)]
        np.save(out_path, np.stack(toks).astype(np.float32))
        return uid, None
    except Exception as e:  # noqa: BLE001
        return uid, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cmvn", choices=["full", "ema"], default="full",
                    help="ema 时特征写入 feats_ema/ (流式因果 CMVN), emb 复用跳过")
    ap.add_argument("--tokens", action="store_true",
                    help="只算多帧 enrollment tokens -> emb_tokens/<id>.npy [N,192]")
    args = ap.parse_args()

    d = Path(args.dir).resolve()
    recs = load_labels(d / "labels.jsonl")
    if args.tokens:
        (d / "emb_tokens").mkdir(exist_ok=True)
        tasks = [(r["id"], r["enrollment"],
                  str(d / "emb_tokens" / f"{r['id']}.npy")) for r in recs]
        print(f"{d.name}: {len(tasks)} 条待算 enrollment tokens")
        errs = done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for uid, err in ex.map(process_tokens, tasks, chunksize=16):
                done += 1
                if err:
                    errs += 1
                    if errs <= 5:
                        print(f"  [错误] {uid}: {err}")
                if done % 2000 == 0:
                    print(f"  进度 {done}/{len(tasks)}")
        print(f"完成 {done - errs}/{len(tasks)}, 失败 {errs}")
        return 1 if errs else 0

    feat_dir = "feats_ema" if args.cmvn == "ema" else "feats"
    (d / feat_dir).mkdir(exist_ok=True)
    (d / "emb").mkdir(exist_ok=True)
    tasks = [(r["id"], r["path"], r["enrollment"],
              str(d / feat_dir / f"{r['id']}.npy"),
              str(d / "emb" / f"{r['id']}.npy"), args.cmvn) for r in recs]
    print(f"{d.name}: {len(tasks)} 条待预计算")
    errs = 0
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for uid, err in ex.map(process_one, tasks, chunksize=16):
            done += 1
            if err:
                errs += 1
                if errs <= 5:
                    print(f"  [错误] {uid}: {err}")
            if done % 500 == 0:
                print(f"  进度 {done}/{len(tasks)}")
    print(f"完成 {done - errs}/{len(tasks)}, 失败 {errs}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
