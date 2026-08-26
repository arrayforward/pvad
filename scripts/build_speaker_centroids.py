# -*- coding: utf-8 -*-
"""计算训练池说话人质心 (CAM++ embedding 每人前 3 条 >=3s 语句平均) 和 top5 近邻。

用法: python scripts/build_speaker_centroids.py [--workers 12]
输出: data/speaker_centroids.npz, data/speaker_top5.json
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvad_common import ROOT, read_wav  # noqa: E402

_BY_SPK = None
_EMB = None


def _init_worker():
    global _BY_SPK
    if _BY_SPK is None:
        split = json.load(open(ROOT / "data" / "split.json", encoding="utf-8"))
        # 全部池 (train/val/test), 供各 split 的易混淆负样本查近邻
        all_spk = set(split["train"]) | set(split["val"]) | set(split["test"])
        _BY_SPK = {}
        with open(ROOT / "data" / "manifest.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["speaker"] in all_spk and r["duration_s"] >= 3.0:
                    _BY_SPK.setdefault(r["speaker"], []).append(r["path"])


def work(spk):
    global _EMB
    _init_worker()
    if _EMB is None:
        from pvad_common import CampplusEmbedder
        _EMB = CampplusEmbedder(intra_threads=1)
    embs = []
    for p in _BY_SPK[spk][:3]:
        try:
            pcm, _ = read_wav(ROOT / p)
            embs.append(_EMB.embed(pcm))
        except Exception:
            pass
    if not embs:
        return spk, None
    c = np.mean(embs, axis=0)
    c /= np.linalg.norm(c) + 1e-10
    return spk, c.astype(np.float32)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    _init_worker()
    spk_list = sorted(_BY_SPK)
    print("训练池说话人:", len(spk_list))
    t0 = time.time()
    centroids = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for spk, c in ex.map(work, spk_list, chunksize=8):
            if c is not None:
                centroids[spk] = c
    np.savez(ROOT / "data" / "speaker_centroids.npz", **centroids)
    print(f"质心 {len(centroids)} 个, 耗时 {time.time()-t0:.0f}s")

    spks = sorted(centroids)
    M = np.stack([centroids[s] for s in spks])
    sims = M @ M.T
    np.fill_diagonal(sims, -2)
    top5 = np.argsort(-sims, axis=1)[:, :5]
    nn = {spks[i]: [spks[j] for j in top5[i]] for i in range(len(spks))}
    with open(ROOT / "data" / "speaker_top5.json", "w", encoding="utf-8") as f:
        json.dump(nn, f, ensure_ascii=False)
    s0 = spks[0]
    idx = {s: i for i, s in enumerate(spks)}
    print("示例:", s0, "->", nn[s0][:3],
          "cos:", [round(float(sims[idx[s0], idx[j]]), 3) for j in nn[s0][:3]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
