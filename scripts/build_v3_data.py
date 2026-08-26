# -*- coding: utf-8 -*-
"""构建 v3 训练数据: v1(干净) + v2(增广) 各 50% 混合, 硬链接复用 wav/特征/embedding,
50% 样本的 enrollment embedding 用 RIR 卷积后的注册语音重算 (混响漂移鲁棒)。

用法: python scripts/build_v3_data.py [--enroll-rir-prob 0.5]
输出: data/mixtures_v3/{train,val,test}/labels.jsonl + 硬链接的 wav/feats/emb
硬负例 (双干扰) 由 gen_mixtures.py --double-interferer --append 之后追加。
"""
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvad_common import ROOT, load_labels, read_wav  # noqa: E402

SPLITS = {"train": 19000, "val": 2500, "test": 2500}
HARD_NEG = {"train": 700, "val": 90, "test": 90}  # 预留给双干扰硬负例的配额


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--enroll-rir-prob", type=float, default=0.5)
    ap.add_argument("--rir-dir", default=str(ROOT / "data" / "raw" / "aug"
                                             / "RIRS_NOISES" / "simulated_rirs"))
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    embedder = None
    rir_files = None

    def enroll_emb(rec, out_path):
        """50% 概率: enrollment 卷随机 RIR 后重算 embedding; 否则硬链接原 emb。"""
        nonlocal embedder, rir_files
        if rng.random() >= args.enroll_rir_prob:
            return False
        if embedder is None:
            from pvad_common import CampplusEmbedder
            embedder = CampplusEmbedder(intra_threads=4)
            rir_files = sorted(Path(args.rir_dir).rglob("*.wav"))
        pcm, _ = read_wav(ROOT / rec["enrollment"])
        rp = rng.choice(rir_files)
        rir, _ = read_wav(rp)
        peak = np.max(np.abs(rir))
        if peak > 1e-8:
            rir = rir / peak
        from scipy.signal import fftconvolve
        wet = fftconvolve(pcm, rir)[: len(pcm)].astype("float32")
        wet *= (np.sqrt((pcm ** 2).mean()) / (np.sqrt((wet ** 2).mean()) + 1e-10))
        emb = embedder.embed(wet)
        np.save(out_path, emb.astype(np.float32))
        return True

    for split, total in SPLITS.items():
        out_dir = ROOT / "data" / "mixtures_v3" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "feats").mkdir(exist_ok=True)
        (out_dir / "emb").mkdir(exist_ok=True)
        n_base = total - HARD_NEG[split]
        per_src = n_base // 2
        merged = []
        for src_name, src_dir in (("v1", ROOT / "data" / "mixtures" / split),
                                  ("v2", ROOT / "data" / "mixtures_v2" / split)):
            recs = load_labels(src_dir / "labels.jsonl")
            rng.shuffle(recs)
            for r in recs[:per_src]:
                nid = f"{src_name}_{r['id']}"
                link(ROOT / r["path"], out_dir / f"{nid}.wav")
                link(src_dir / "feats" / f"{r['id']}.npy",
                     out_dir / "feats" / f"{nid}.npy")
                emb_out = out_dir / "emb" / f"{nid}.npy"
                did_rir = enroll_emb(r, emb_out)
                if not did_rir:
                    link(src_dir / "emb" / f"{r['id']}.npy", emb_out)
                nr = dict(r)
                nr["id"] = nid
                nr["path"] = f"data/mixtures_v3/{split}/{nid}.wav"
                nr["src"] = src_name
                nr["enroll_rir"] = did_rir
                merged.append(nr)
        rng.shuffle(merged)
        with open(out_dir / "labels.jsonl", "w", encoding="utf-8") as f:
            for nr in merged:
                f.write(json.dumps(nr, ensure_ascii=False) + "\n")
        n_neg = sum(1 for r in merged if r.get("negative"))
        n_rir = sum(1 for r in merged if r.get("enroll_rir"))
        print(f"[{split}] base {len(merged)} 条 (负样本 {n_neg}, "
              f"enrollment-RIR {n_rir}), 预留硬负例 {HARD_NEG[split]}")
    print("完成。接下来运行 gen_mixtures --double-interferer --append 补硬负例, "
          "再 precompute_features。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
