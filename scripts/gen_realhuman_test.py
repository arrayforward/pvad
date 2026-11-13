# -*- coding: utf-8 -*-
"""真人 TSE 验收测试集生成：100 条，全部来自 split.json test 池真人语音。

场景与电平按实际部署（A 近场大、B 远场小，替代已删除的错误 mixtures/test）：
  - 70 条 A+B：A 有效 RMS 归一 0.03~0.06，B 低 5~15dB，注册 A 的其它 3-10s 句子
  - 30 条 B_only：只有 B（与混合中同等低电平），注册不在场的 A；target.wav 全零

每条落盘 {id}_mix.wav / {id}_target.wav（混合前干净 A 轨）/ {id}_enroll.wav，
meta.jsonl 记录说话人、SNR、实测电平与 10ms 帧标签（0 静音 1 干扰 2 目标）。

用法:
  python scripts/gen_realhuman_test.py --out D:/ai-app/pvad-data/real_test_tse
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

DV = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DV / "scripts"))
from pvad_common import DATA_ROOT  # noqa: E402
from gen_mixtures import build_track, frame_labels, load_manifest, read_wav, rms  # noqa: E402

SR = 16000


def active_rms(x, mask):
    seg = x[mask == 1]
    return rms(seg) if len(seg) else rms(x)


def pick_enroll(rng, utts, exclude):
    cands = [u for u in utts if 3.0 <= u[1] <= 10.0 and str(u[0]) != exclude]
    return rng.choice(cands)[0] if cands else None


def write_wav(path, x):
    sf.write(str(path), np.asarray(x, dtype=np.float32), SR, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_ROOT / "real_test_tse"))
    ap.add_argument("--n-mix", type=int, default=70)
    ap.add_argument("--n-bonly", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261112)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    by_spk = load_manifest(DATA_ROOT / "manifest.jsonl")
    with open(DATA_ROOT / "split.json", encoding="utf-8") as f:
        pools = json.load(f)
    speakers = [s for s in pools["test"] if s in by_spk and len(by_spk[s]) >= 3]
    print(f"test 池真人说话人 {len(speakers)}")

    rows = []
    # ---- A+B 正样本 ----
    made = 0
    while made < args.n_mix:
        tgt_spk, itf_spk = rng.sample(speakers, 2)
        length = int(rng.uniform(3.0, 8.0) * SR)
        tgt, mask_t, _ = build_track(rng, by_spk[tgt_spk], length)
        itf, mask_g, _ = build_track(rng, by_spk[itf_spk], length)
        if not (mask_t == 1).any() or not (mask_g == 1).any():
            continue
        enroll = pick_enroll(rng, by_spk[tgt_spk], None)
        if enroll is None:
            continue
        level = float(np.clip(rng.gauss(0.045, 0.008), 0.03, 0.06))
        snr_db = rng.uniform(5.0, 15.0)
        a_r, b_r = active_rms(tgt, mask_t), active_rms(itf, mask_g)
        ga = level / a_r
        gb = ga * a_r / b_r / 10 ** (snr_db / 20.0)
        mix = np.clip(tgt * ga + itf * gb, -0.98, 0.98).astype(np.float32)
        uid = f"rt_{made:04d}"
        write_wav(out / f"{uid}_mix.wav", mix)
        write_wav(out / f"{uid}_target.wav", tgt * ga)  # 干净 A 轨（混合前同尺度）
        write_wav(out / f"{uid}_enroll.wav", read_wav(enroll))
        rows.append({"id": uid, "scenario": "A_mix", "target_speaker": tgt_spk,
                     "interferer_speaker": itf_spk, "snr_db": round(snr_db, 2),
                     "a_rms": round(a_r * ga, 5), "b_rms": round(b_r * gb, 5),
                     "duration_s": round(length / SR, 3),
                     "labels": frame_labels(mask_t, mask_g)})
        made += 1

    # ---- B_only 负样本 ----
    made_b = 0
    while made_b < args.n_bonly:
        enroll_spk, itf_spk = rng.sample(speakers, 2)
        length = int(rng.uniform(3.0, 8.0) * SR)
        itf, mask_g, _ = build_track(rng, by_spk[itf_spk], length)
        if not (mask_g == 1).any():
            continue
        enroll = pick_enroll(rng, by_spk[enroll_spk], None)
        if enroll is None:
            continue
        # B 电平与混合场景中一致：近场 level 再低 5~15dB
        level = float(np.clip(rng.gauss(0.045, 0.008), 0.03, 0.06))
        snr_db = rng.uniform(5.0, 15.0)
        gb = level / 10 ** (snr_db / 20.0) / active_rms(itf, mask_g)
        mix = np.clip(itf * gb, -0.98, 0.98).astype(np.float32)
        uid = f"rt_{args.n_mix + made_b:04d}"
        write_wav(out / f"{uid}_mix.wav", mix)
        write_wav(out / f"{uid}_target.wav", np.zeros(length, dtype=np.float32))
        write_wav(out / f"{uid}_enroll.wav", read_wav(enroll))
        rows.append({"id": uid, "scenario": "B_only", "target_speaker": enroll_spk,
                     "interferer_speaker": itf_spk, "snr_db": round(snr_db, 2),
                     "a_rms": 0.0, "b_rms": round(active_rms(itf, mask_g) * gb, 5),
                     "duration_s": round(length / SR, 3),
                     "labels": frame_labels(np.zeros(length, np.int8), mask_g)})
        made_b += 1

    with open(out / "meta.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    am = [r for r in rows if r["scenario"] == "A_mix"]
    print(f"完成 {len(rows)} 条 -> {out}")
    print(f"电平审计: A RMS {min(r['a_rms'] for r in am):.4f}~{max(r['a_rms'] for r in am):.4f}, "
          f"B RMS {min(r['b_rms'] for r in rows):.4f}~{max(r['b_rms'] for r in rows):.4f}, "
          f"SNR {min(r['snr_db'] for r in am):.1f}~{max(r['snr_db'] for r in am):.1f}dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
