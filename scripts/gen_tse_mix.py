# -*- coding: utf-8 -*-
"""TSE 数据集生成器 (data/tse_mix/)：三元组 mix.wav / target.wav / interferer.wav + meta。

分布沿用 mixtures_v4：50% 干净 / 50% 增广（目标/干扰各 70% 独立卷 RIR，
60% 叠加 MUSAN 噪声 SNR 5-20dB），干扰人 20% 取目标音色 top5 近邻，
20% target-absent 负样本（target.wav 全零，教"输出静音"），
enrollment 规则同 v4（目标说话人其他句子 3-10s，不混入）。

target.wav = 混合前目标语音的原始干净波形（无干扰、无 RIR、无噪声、
未做 SNR 缩放；SI-SNR 评估天然尺度不变）。

用法:
  python scripts/gen_tse_mix.py --split train --n 15000 --seed 8001
  python scripts/gen_tse_mix.py --split val --n 1000 --seed 8002
  python scripts/gen_tse_mix.py --split test --n 1000 --seed 8003
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_mixtures import Augmenter, build_track, load_manifest, rms  # noqa: E402
from pvad_common import ROOT, DATA_ROOT, manifest_rel  # noqa: E402

SR = 16000
RIR_DIR = DATA_ROOT / "raw" / "aug" / "RIRS_NOISES" / "simulated_rirs"
NOISE_DIR = DATA_ROOT / "raw" / "aug" / "musan"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=8001)
    ap.add_argument("--neg-ratio", type=float, default=0.2)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = DATA_ROOT / "tse_mix" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    aug = Augmenter(str(RIR_DIR), str(NOISE_DIR))

    with open(DATA_ROOT / "split.json", encoding="utf-8") as f:
        pools = json.load(f)
    with open(DATA_ROOT / "speaker_top5.json", encoding="utf-8") as f:
        top5 = json.load(f)
    by_spk = load_manifest(DATA_ROOT / "manifest.jsonl")
    speakers = [s for s in pools[args.split] if s in by_spk and len(by_spk[s]) >= 3]
    print(f"[{args.split}] 说话人池 {len(speakers)}, n={args.n}")

    made = 0
    metas = []
    labels_rows = []
    while made < args.n:
        uid = f"tse_{made:05d}"
        tgt_spk = rng.choice(speakers)
        if rng.random() < 0.2 and tgt_spk in top5:
            itf_spk = rng.choice(top5[tgt_spk])
        else:
            itf_spk = rng.choice(speakers)
            while itf_spk == tgt_spk:
                itf_spk = rng.choice(speakers)
        neg = rng.random() < args.neg_ratio

        L = rng.uniform(3.0, 8.0)
        n = int(L * SR)
        tgt, mask_t, used_t = build_track(rng, by_spk[tgt_spk], n)
        itf, mask_g, _ = build_track(rng, by_spk[itf_spk], n)
        if not (mask_g == 1).any() or (not neg and not (mask_t == 1).any()):
            continue
        clean_target = tgt.copy()  # 原始干净目标轨 (未增广未缩放)

        # 样本级 50/50 干净/增广门控 (与 mixtures_v4 同分布);
        # 增广样本内: 目标/干扰各 70% 独立卷 RIR, 60% 叠加 MUSAN 噪声
        is_aug = rng.random() < 0.5
        tgt_m, itf_m = tgt, itf
        if is_aug and aug.rirs and rng.random() < 0.7:
            tgt_m = aug.apply_rir(tgt_m, rng)
        if is_aug and aug.rirs and rng.random() < 0.7:
            itf_m = aug.apply_rir(itf_m, rng)

        snr_db = rng.uniform(-5.0, 10.0)
        s_t, s_g = tgt_m[mask_t == 1], itf_m[mask_g == 1]
        if len(s_t) == 0 or len(s_g) == 0:
            continue
        k = rms(s_t) / (rms(s_g) * 10 ** (snr_db / 20.0))
        mix = tgt_m + itf_m * k
        itf_out = itf_m * k  # 与混合时同增益的干扰轨 (评估参考)
        if is_aug and aug.noises and rng.random() < 0.6:
            mix = aug.add_noise(mix, rng)
        peak = np.max(np.abs(mix)) + 1e-10
        if peak > 0.99:
            mix = mix / peak * 0.99

        # enrollment: 目标说话人其他句子 3-10s
        used_set = set(used_t)
        cands = [p for p, d in by_spk[tgt_spk]
                 if 3.0 <= d <= 10.0 and str(p) not in used_set]
        if not cands:
            continue
        enroll = rng.choice(cands)

        sf.write(str(out_dir / f"{uid}_mix.wav"), mix, SR, subtype="PCM_16")
        sf.write(str(out_dir / f"{uid}_target.wav"),
                 np.zeros(n, dtype="float32") if neg else clean_target, SR,
                 subtype="PCM_16")
        sf.write(str(out_dir / f"{uid}_interferer.wav"), itf_out, SR,
                 subtype="PCM_16")
        metas.append({
            "id": uid,
            "target_speaker": tgt_spk,
            "interferer_speaker": itf_spk,
            "snr_db": round(snr_db, 2),
            "duration_s": round(n / SR, 3),
            "negative": neg,
            "augmented": is_aug,
            "enrollment": manifest_rel(enroll),
        })
        # precompute_features 兼容行 (emb 预计算用)
        labels_rows.append({
            "id": uid,
            "path": f"data/tse_mix/{args.split}/{uid}_mix.wav",
            "enrollment": manifest_rel(enroll),
            "duration_s": round(n / SR, 3),
        })
        made += 1
        if made % 1000 == 0:
            print(f"  已生成 {made}/{args.n}")

    with open(out_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(out_dir / "labels.jsonl", "w", encoding="utf-8") as f:
        for m in labels_rows:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    n_neg = sum(1 for m in metas if m["negative"])
    n_aug = sum(1 for m in metas if m["augmented"])
    print(f"完成: {out_dir} {made} 条 (负样本 {n_neg}, 增广 {n_aug})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
