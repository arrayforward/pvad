# -*- coding: utf-8 -*-
"""长音频流数据组装器 (data/mixtures_long/)：30-90s 连续流 + 帧级三分类标签。

事件调度：0-10s 噪声/静音开场 → 随机事件序列（目标说话 3-10s / 非目标连续说话
5-30s / TTS 回声(-6~-12dB+RIR) / 纯噪声 5-20s / 双讲重叠(SNR -5~10dB)），
事件间隔 0.3-3s 静音。标签由拼接位置精确生成（0 静音 / 1 非目标 / 2 目标，重叠计 2）。

配额（每 split）：short 20%（3-8s 原分布）/ pure_neg 20%（全程无目标）/
absorb 30%（≥15s 连续非目标后接目标）/ normal 30%。
干扰人 20% 概率取目标说话人音色 top5 近邻（data/speaker_top5.json）。
enrollment：目标说话人其他句子 3-10s（不混入流），50% 卷 RIR 后另存。

用法:
  python scripts/gen_longstream.py --split train --n 3500 --seed 7001
  python scripts/gen_longstream.py --split val --n 300 --seed 7002
  python scripts/gen_longstream.py --split test --n 300 --seed 7003
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_mixtures import Augmenter, frame_labels, load_manifest, read_wav, rms  # noqa: E402
from pvad_common import ROOT  # noqa: E402

SR = 16000
FRAME = 160
RIR_DIR = ROOT / "data" / "raw" / "aug" / "RIRS_NOISES" / "simulated_rirs"
NOISE_DIR = ROOT / "data" / "raw" / "aug" / "musan"


def speech_track(rng, utts, dur_s):
    """拼接某说话人语句填充 dur_s, 返回 (track, mask)。"""
    n = int(dur_s * SR)
    track = np.zeros(n, dtype="float32")
    mask = np.zeros(n, dtype=np.int8)
    pool = utts[:]
    rng.shuffle(pool)
    pos, idx = 0, 0
    while pos < n - SR // 4:
        utt = read_wav(pool[idx % len(pool)][0])
        idx += 1
        if len(utt) < SR // 8:
            continue
        end = min(pos + len(utt), n)
        track[pos:end] += utt[: end - pos]
        mask[pos:end] = 1
        pos = end + rng.randint(int(0.1 * SR), int(0.4 * SR))
    return track, mask


def noise_track(rng, aug, dur_s, lo=0.02, hi=0.08):
    n = int(dur_s * SR)
    src = aug._load(rng.choice(aug.noises))
    if len(src) < n:
        src = np.tile(src, n // len(src) + 1)
    off = rng.randint(0, len(src) - n)
    return src[off: off + n] * rng.uniform(lo, hi) / (rms(src) + 1e-10) * 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=7001)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out).resolve() if args.out else (
        ROOT / "data" / "mixtures_long" / args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "enrollments").mkdir(exist_ok=True)
    aug = Augmenter(str(RIR_DIR), str(NOISE_DIR))

    with open(ROOT / "data" / "split.json", encoding="utf-8") as f:
        pools = json.load(f)
    with open(ROOT / "data" / "speaker_top5.json", encoding="utf-8") as f:
        top5 = json.load(f)
    by_spk = load_manifest(ROOT / "data" / "manifest.jsonl")
    speakers = [s for s in pools[args.split] if s in by_spk and len(by_spk[s]) >= 3]
    print(f"[{args.split}] 说话人池 {len(speakers)}, 输出 {out_dir}")

    def pick_pair():
        tgt = rng.choice(speakers)
        if rng.random() < 0.2 and tgt in top5:
            itf = rng.choice(top5[tgt])
        else:
            itf = rng.choice(speakers)
            while itf == tgt:
                itf = rng.choice(speakers)
        return tgt, itf

    def rir_maybe(track, p=0.5):
        return aug.apply_rir(track, rng) if (aug.rirs and rng.random() < p) else track

    def make_enrollment(tgt_spk, used, uid):
        cands = [p for p, d in by_spk[tgt_spk]
                 if 3.0 <= d <= 10.0 and str(p) not in used]
        if not cands:
            cands = [p for p, d in by_spk[tgt_spk] if 2.0 <= d <= 10.0]
        src = rng.choice(cands)
        if rng.random() < 0.5 and aug.rirs:  # 50% enrollment 卷 RIR
            pcm = read_wav(src)
            wet = aug.apply_rir(pcm, rng)
            ep = out_dir / "enrollments" / f"{uid}.wav"
            sf.write(str(ep), wet, SR, subtype="PCM_16")
            return ep.relative_to(ROOT).as_posix()
        return str(src.relative_to(ROOT)).replace("\\", "/")

    kinds = (["short"] * 20 + ["pure_neg"] * 20 + ["absorb"] * 30 + ["normal"] * 30)
    cycle = kinds[:]
    rng.shuffle(cycle)  # 每 100 条精确 20/20/30/30 配额
    labels_path = out_dir / "labels.jsonl"
    with open(labels_path, "w", encoding="utf-8") as lf:
        for i in range(args.n):
            kind = cycle[i % 100]
            uid = f"ls_{i:05d}"
            used_src = []

            if kind == "short":
                # 3-8s 原分布: 目标+干扰 (或纯负)
                tgt_spk, itf_spk = pick_pair()
                L = rng.uniform(3.0, 8.0)
                n = int(L * SR)
                neg = rng.random() < 0.25
                tg, mg = speech_track(rng, by_spk[itf_spk], L)
                used_src.append(str(by_spk[itf_spk][0][0]))
                if neg:
                    mix, mask_t = tg, np.zeros(n, dtype=np.int8)
                else:
                    tt, mask_t = speech_track(rng, by_spk[tgt_spk], L)
                    s_t, s_g = tt[mask_t == 1], tg[mg == 1]
                    k = rms(s_t) / (rms(s_g) * 10 ** (rng.uniform(-5, 10) / 20.0))
                    mix = rir_maybe(tt) + rir_maybe(tg) * k
                mask_g = mg
            else:
                tgt_spk, itf_spk = pick_pair()
                L = rng.uniform(30.0, 80.0)
                n = int(L * SR)
                mix = np.zeros(n, dtype="float32")
                mask_t = np.zeros(n, dtype=np.int8)
                mask_g = np.zeros(n, dtype=np.int8)
                pos = 0
                # 开场 0-10s 噪声或静音
                o = rng.uniform(0, 10)
                if o > 0.5:
                    seg = noise_track(rng, aug, o) if rng.random() < 0.5 else np.zeros(int(o * SR), dtype="float32")
                    mix[pos: pos + len(seg)] += seg
                    pos += len(seg)
                # absorb: 先排一个 >=15s 非目标长事件
                if kind == "absorb":
                    d = rng.uniform(15, 30)
                    # 保证非目标段后还有空间放 gap+目标段
                    d = min(d, (n - pos) / SR - 13.0)
                    if d < 10:  # 空间不足就当 normal 处理
                        kind = "normal"
                if kind == "absorb":
                    tr, mk = speech_track(rng, by_spk[itf_spk], d)
                    tr = rir_maybe(tr)
                    e = min(pos + len(tr), n)
                    mix[pos:e] += tr[: e - pos]
                    mask_g[pos:e] = mk[: e - pos]
                    pos = e + int(rng.uniform(0.3, 3) * SR)
                    # 接目标 (吸收态正解场景)
                    d2 = rng.uniform(3, 10)
                    tr2, mk2 = speech_track(rng, by_spk[tgt_spk], d2)
                    tr2 = rir_maybe(tr2)
                    e = min(pos + len(tr2), n)
                    mix[pos:e] += tr2[: e - pos]
                    mask_t[pos:e] = mk2[: e - pos]
                    pos = e + int(rng.uniform(0.3, 3) * SR)
                # 随机事件填满
                while pos < n - 4 * SR:
                    ev = rng.choice(["tgt", "nontgt", "echo", "noise", "overlap", "gap"])
                    if kind == "pure_neg" and ev in ("tgt", "overlap"):
                        ev = "nontgt"
                    if ev == "gap":
                        pos += int(rng.uniform(0.3, 3) * SR)
                        continue
                    d = {"tgt": rng.uniform(3, 10), "nontgt": rng.uniform(5, 30),
                         "echo": rng.uniform(5, 15), "noise": rng.uniform(5, 20),
                         "overlap": rng.uniform(3, 8)}[ev]
                    d = min(d, (n - pos) / SR - 1)
                    if d < 2:
                        break
                    if ev == "noise":
                        seg = noise_track(rng, aug, d)
                        mix[pos: pos + len(seg)] += seg
                        pos += len(seg)
                    elif ev == "echo":
                        tr, mk = speech_track(rng, by_spk[itf_spk], d)
                        tr = aug.apply_rir(tr, rng) * (10 ** (rng.uniform(-12, -6) / 20.0))
                        e = min(pos + len(tr), n)
                        mix[pos:e] += tr[: e - pos]
                        mask_g[pos:e] = mk[: e - pos]
                        pos = e
                    elif ev == "overlap":
                        tt, mk_t = speech_track(rng, by_spk[tgt_spk], d)
                        tg, mk_g = speech_track(rng, by_spk[itf_spk], d)
                        k = rms(tt[mk_t == 1]) / (rms(tg[mk_g == 1]) * 10 ** (rng.uniform(-5, 10) / 20.0))
                        tr = rir_maybe(tt) + rir_maybe(tg) * k
                        e = min(pos + len(tr), n)
                        mix[pos:e] += tr[: e - pos]
                        mask_t[pos:e] = mk_t[: e - pos]
                        mask_g[pos:e] = np.maximum(mask_g[pos:e], mk_g[: e - pos])
                        pos = e
                    else:  # tgt / nontgt
                        spk = tgt_spk if ev == "tgt" else itf_spk
                        tr, mk = speech_track(rng, by_spk[spk], d)
                        tr = rir_maybe(tr)
                        e = min(pos + len(tr), n)
                        mix[pos:e] += tr[: e - pos]
                        if ev == "tgt":
                            mask_t[pos:e] = mk[: e - pos]
                        else:
                            mask_g[pos:e] = mk[: e - pos]
                        pos = e
                    pos += int(rng.uniform(0.3, 3) * SR)
                n = min(n, pos)
                mix, mask_t, mask_g = mix[:n], mask_t[:n], mask_g[:n]

            peak = np.max(np.abs(mix)) + 1e-10
            if peak > 0.99:
                mix = mix / peak * 0.99
            wav_path = out_dir / f"{uid}.wav"
            sf.write(str(wav_path), mix, SR, subtype="PCM_16")
            lab = frame_labels(mask_t, mask_g)
            enroll = make_enrollment(tgt_spk, used_src, uid)
            lf.write(json.dumps({
                "id": uid,
                "path": wav_path.relative_to(ROOT).as_posix(),
                "target_speaker": tgt_spk,
                "interferer_speaker": itf_spk,
                "snr_db": round(rng.uniform(-5, 10), 2),
                "duration_s": round(len(mix) / SR, 3),
                "frame_ms": 10,
                "enrollment": enroll,
                "negative": kind == "pure_neg" or (kind == "short" and neg),
                "kind": kind,
                "overlap_frames": [],
                "labels": lab,
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  已生成 {i + 1}/{args.n}")
    print(f"完成: {labels_path} ({args.n} 条)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
