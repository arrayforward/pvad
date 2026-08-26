# -*- coding: utf-8 -*-
"""PVAD 训练样本合成: 目标说话人 + 干扰说话人混合, 帧级三分类标签。

用法:
    python scripts/gen_mixtures.py --n 50 --out data/mixtures/sample
    python scripts/gen_mixtures.py --n 10000 --out data/mixtures/train --seed 1

每条样本:
  - 长度 3-8s, 16kHz 单声道 PCM16
  - 目标说话人语音 + 另一说话人干扰语音, SNR 随机 -5~10dB
  - 50% 概率叠加整体增益扰动 (±6dB) 和随机带通 EQ
标签 (labels.jsonl, 10ms 帧):
  0=静音  1=非目标(干扰)语音  2=目标语音
标签由拼接位置直接生成 (非 VAD 估计), 精确到样本级再聚合到帧。
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SR = 16000
FRAME_MS = 10
FRAME = SR * FRAME_MS // 1000  # 160 samples


def load_manifest(path: Path):
    by_spk = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_spk[r["speaker"]].append((ROOT / r["path"], r["duration_s"]))
    # 过滤文件不存在的
    return {s: [pd for pd in ps if pd[0].exists()] for s, ps in by_spk.items()}


def read_wav(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        n = int(round(len(data) * SR / sr))
        x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
        data = np.interp(x_new, x_old, data[:, 0]).astype("float32")[:, None]
    return data[:, 0] if data.shape[1] > 0 else np.zeros(0, dtype="float32")


def build_track(rng: random.Random, utts: list, length: int):
    """从说话人的语句列表 [(path, dur)] 拼接填充 length 采样点。
    返回 (track, mask, used): mask[i]=1 表示语音; used 为实际混入的源文件路径集合。"""
    track = np.zeros(length, dtype="float32")
    mask = np.zeros(length, dtype=np.int8)
    used = []
    pos = rng.randint(0, min(SR // 2, length // 4))  # 随机起始偏移
    pool = utts[:]
    rng.shuffle(pool)
    idx = 0
    while pos < length - SR // 4:
        utt = read_wav(pool[idx % len(pool)][0])
        if len(utt) < SR // 8:
            idx += 1
            continue
        used.append(str(pool[idx % len(pool)][0]))
        idx += 1
        end = min(pos + len(utt), length)
        seg = utt[: end - pos]
        track[pos:end] += seg
        mask[pos:end] = 1
        gap = rng.randint(int(0.1 * SR), int(0.6 * SR))  # 语句间静音间隙
        pos = end + gap
    return track, mask, used


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-10)


def random_bandpass(x: np.ndarray, rng: random.Random) -> np.ndarray:
    """随机带通: FFT 域衰减带外成分, 模拟简单 EQ 扰动。"""
    n = len(x)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    lo = rng.uniform(50, 400)
    hi = rng.uniform(3000, 7800)
    if hi <= lo + 500:
        hi = lo + 500
    gain = np.ones_like(freqs)
    # 带外衰减 (平滑过渡, 不做硬截断)
    gain *= 1.0 / (1.0 + (lo / np.maximum(freqs, 1.0)) ** 4) if lo > 60 else 1.0
    gain *= 1.0 / (1.0 + (np.maximum(freqs, 1.0) / hi) ** 8)
    gain = np.clip(gain, 0.05, 1.0)
    return np.fft.irfft(spec * gain, n=n).astype("float32")


def frame_labels(mask_target: np.ndarray, mask_interf: np.ndarray) -> list:
    """采样级 mask 聚合到 10ms 帧。重叠时目标优先 (2 > 1 > 0)。"""
    n_frames = (len(mask_target) + FRAME - 1) // FRAME
    labels = []
    for i in range(n_frames):
        s, e = i * FRAME, min((i + 1) * FRAME, len(mask_target))
        t = mask_target[s:e].mean()
        g = mask_interf[s:e].mean()
        if t >= 0.5:
            labels.append(2)
        elif g >= 0.5:
            labels.append(1)
        else:
            labels.append(0)
    return labels


# ---------------- 增广: RIR 卷积 + MUSAN 噪声 ----------------

class Augmenter:
    """RIR 混响 + MUSAN 噪声增广。标签语义不变 (噪声不计为任何类)。"""

    def __init__(self, rir_dir=None, noise_dir=None):
        from scipy.signal import fftconvolve
        self._fftconvolve = fftconvolve
        self.rirs = sorted(Path(rir_dir).rglob("*.wav")) if rir_dir else []
        self.noises = []
        if noise_dir:
            nd = Path(noise_dir)
            for sub in ("noise", "music"):
                if (nd / sub).is_dir():
                    self.noises += sorted((nd / sub).rglob("*.wav"))
            if not self.noises:  # 没有子目录结构就全目录找
                self.noises = sorted(nd.rglob("*.wav"))
        self._rir_cache = {}
        print(f"增广: RIR {len(self.rirs)} 个, 噪声 {len(self.noises)} 个")

    def _load(self, path):
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if data.shape[1] > 1:
            data = data.mean(axis=1, keepdims=True)
        data = data[:, 0]
        if sr != SR:
            n = int(round(len(data) * SR / sr))
            x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
            data = np.interp(x_new, x_old, data).astype("float32")
        return data

    def apply_rir(self, x: np.ndarray, rng: random.Random) -> np.ndarray:
        """随机 RIR 卷积, 归一化到原 RMS。"""
        p = rng.choice(self.rirs)
        key = str(p)
        if key not in self._rir_cache:
            rir = self._load(p)
            peak = np.max(np.abs(rir))
            if peak > 1e-8:
                rir = rir / peak
            self._rir_cache[key] = rir
        rir = self._rir_cache[key]
        y = self._fftconvolve(x, rir)[: len(x)].astype("float32")
        return y * (rms(x) / rms(y))

    def add_noise(self, x: np.ndarray, rng: random.Random) -> np.ndarray:
        """叠加 MUSAN 噪声, SNR 5-20dB (相对 x 的 RMS)。"""
        noise = self._load(rng.choice(self.noises))
        n = len(x)
        if len(noise) < n:  # 循环补齐
            reps = n // len(noise) + 1
            noise = np.tile(noise, reps)
        off = rng.randint(0, len(noise) - n)
        noise = noise[off: off + n]
        snr_db = rng.uniform(5.0, 20.0)
        scale = rms(x) / (rms(noise) * 10 ** (snr_db / 20.0))
        return (x + noise * scale).astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "mixtures" / "sample"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool", choices=["train", "val", "test"], default=None,
                    help="只从 data/split.json 对应说话人池抽取")
    ap.add_argument("--split-file", default=str(ROOT / "data" / "split.json"))
    ap.add_argument("--negatives", type=int, default=0,
                    help="追加生成 target-absent 负样本 (只有干扰说话人, "
                         "enrollment 来自不在场的另一说话人, 标签无 2)")
    ap.add_argument("--double-interferer", action="store_true",
                    help="负样本用两个干扰说话人同时干扰 (硬负例)")
    ap.add_argument("--append", action="store_true",
                    help="追加到已有 labels.jsonl (id 顺延), 而非重写")
    ap.add_argument("--rir-dir", default=None, help="RIR wav 目录 (卷积混响)")
    ap.add_argument("--noise-dir", default=None,
                    help="MUSAN 根目录 (取 noise/ 和 music/ 子集)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    aug = Augmenter(args.rir_dir, args.noise_dir) if (args.rir_dir or args.noise_dir) else None

    by_spk = load_manifest(Path(args.manifest))
    speakers = [s for s, ps in by_spk.items() if len(ps) >= 2]
    if args.pool:
        with open(args.split_file, encoding="utf-8") as f:
            pools = json.load(f)
        allow = set(pools[args.pool])
        speakers = [s for s in speakers if s in allow]
    if len(speakers) < 2:
        print(f"可用说话人不足 ({len(speakers)}), 请先运行 prepare_corpus.py")
        return 1
    print(f"可用说话人 {len(speakers)} 个 (pool={args.pool or 'all'})")

    labels_path = out_dir / "labels.jsonl"
    made = 0
    mode = "w"
    if args.append and labels_path.exists():
        with open(labels_path, encoding="utf-8") as f:
            made = sum(1 for line in f if line.strip())
        mode = "a"
        print(f"append 模式: 已有 {made} 条, id 顺延")
    with open(labels_path, mode, encoding="utf-8") as lf:
        i = 0
        pos_target = made + args.n
        while made < pos_target:
            i += 1
            tgt_spk, itf_spk = rng.sample(speakers, 2)
            dur_s = rng.uniform(3.0, 8.0)
            length = int(dur_s * SR)
            tgt, mask_t, used_t = build_track(rng, by_spk[tgt_spk], length)
            itf, mask_g, _ = build_track(rng, by_spk[itf_spk], length)

            # RIR 增广: 70% 概率, 目标/干扰各自独立卷随机 RIR
            if aug and aug.rirs:
                if rng.random() < 0.7:
                    tgt = aug.apply_rir(tgt, rng)
                if rng.random() < 0.7:
                    itf = aug.apply_rir(itf, rng)

            # SNR 混合: 目标 RMS 相对干扰 RMS
            snr_db = rng.uniform(-5.0, 10.0)
            speech_t = tgt[mask_t == 1]
            speech_g = itf[mask_g == 1]
            if len(speech_t) == 0 or len(speech_g) == 0:
                continue

            # enrollment: 目标说话人的其它句子, 3-10s, 不含已混入的
            used_set = set(used_t)
            cands = [p for p, d in by_spk[tgt_spk]
                     if 3.0 <= d <= 10.0 and str(p) not in used_set]
            if not cands:
                continue
            enroll = rng.choice(cands)

            scale = rms(speech_t) / (rms(speech_g) * 10 ** (snr_db / 20.0))
            mix = tgt + itf * scale

            # 噪声增广: 60% 概率叠加 MUSAN (SNR 5-20dB), 噪声不计为任何类
            if aug and aug.noises and rng.random() < 0.6:
                mix = aug.add_noise(mix, rng)

            # 扰动: 50% 带通, 50% 整体增益
            if rng.random() < 0.5:
                mix = random_bandpass(mix, rng)
            if rng.random() < 0.5:
                mix = mix * (10 ** (rng.uniform(-6, 6) / 20.0))
            peak = np.max(np.abs(mix)) + 1e-10
            if peak > 0.99:
                mix = mix / peak * 0.99

            uid = f"mix_{made:06d}"
            wav_path = out_dir / f"{uid}.wav"
            sf.write(str(wav_path), mix, SR, subtype="PCM_16")
            labels = frame_labels(mask_t, mask_g)
            # 双讲区间 (目标与干扰同时活动的帧), 供端到端评估用
            overlap = []
            n_frames = len(labels)
            both = np.zeros(n_frames, dtype=bool)
            for fi in range(n_frames):
                s, e = fi * FRAME, min((fi + 1) * FRAME, length)
                if mask_t[s:e].mean() >= 0.5 and mask_g[s:e].mean() >= 0.5:
                    both[fi] = True
            st = None
            for fi in range(n_frames + 1):
                if fi < n_frames and both[fi] and st is None:
                    st = fi
                elif (fi >= n_frames or not both[fi]) and st is not None:
                    overlap.append([st, fi - 1])
                    st = None
            lf.write(json.dumps({
                "id": uid,
                "path": str(wav_path.relative_to(ROOT)).replace("\\", "/"),
                "target_speaker": tgt_spk,
                "interferer_speaker": itf_spk,
                "snr_db": round(snr_db, 2),
                "duration_s": round(len(mix) / SR, 3),
                "frame_ms": FRAME_MS,
                "enrollment": str(enroll.relative_to(ROOT)).replace("\\", "/"),
                "overlap_frames": overlap,
                "labels": labels,
            }, ensure_ascii=False) + "\n")
            made += 1
            if made % 500 == 0:
                print(f"已生成 {made}/{pos_target}")

        # ---- target-absent 负样本: 只有干扰说话人, enrollment 是不在场说话人 ----
        neg_target = made + args.negatives
        while made < neg_target:
            if args.double_interferer:
                enroll_spk, itf_spk, itf_spk2 = rng.sample(speakers, 3)
            else:
                enroll_spk, itf_spk = rng.sample(speakers, 2)
                itf_spk2 = None
            dur_s = rng.uniform(3.0, 8.0)
            length = int(dur_s * SR)
            itf, mask_g, _ = build_track(rng, by_spk[itf_spk], length)
            if not (mask_g == 1).any():
                continue
            if itf_spk2:  # 硬负例: 第二个干扰说话人同时干扰
                itf2, mask_g2, _ = build_track(rng, by_spk[itf_spk2], length)
                if not (mask_g2 == 1).any():
                    continue
                rel_db = rng.uniform(-5.0, 5.0)
                s1 = itf[mask_g == 1]
                s2 = itf2[mask_g2 == 1]
                k = rms(s1) / (rms(s2) * 10 ** (rel_db / 20.0))
                itf = itf + itf2 * k
                mask_g = np.maximum(mask_g, mask_g2)
            cands = [p for p, d in by_spk[enroll_spk] if 3.0 <= d <= 10.0]
            if not cands:
                continue
            enroll = rng.choice(cands)
            mix = itf
            # 负样本同样增广: 70% RIR, 60% 噪声
            if aug and aug.rirs and rng.random() < 0.7:
                mix = aug.apply_rir(mix, rng)
            if aug and aug.noises and rng.random() < 0.6:
                mix = aug.add_noise(mix, rng)
            if rng.random() < 0.5:
                mix = random_bandpass(mix, rng)
            if rng.random() < 0.5:
                mix = mix * (10 ** (rng.uniform(-6, 6) / 20.0))
            peak = np.max(np.abs(mix)) + 1e-10
            if peak > 0.99:
                mix = mix / peak * 0.99
            uid = f"mix_{made:06d}"
            wav_path = out_dir / f"{uid}.wav"
            sf.write(str(wav_path), mix, SR, subtype="PCM_16")
            labels = frame_labels(np.zeros(length, dtype=np.int8), mask_g)
            lf.write(json.dumps({
                "id": uid,
                "path": str(wav_path.relative_to(ROOT)).replace("\\", "/"),
                "target_speaker": enroll_spk,
                "interferer_speaker": itf_spk,
                "snr_db": round(rng.uniform(-5.0, 10.0), 2),  # 名义值, 无目标语音
                "duration_s": round(len(mix) / SR, 3),
                "frame_ms": FRAME_MS,
                "enrollment": str(enroll.relative_to(ROOT)).replace("\\", "/"),
                "overlap_frames": [],
                "negative": True,
                "double_interferer": bool(itf_spk2),
                "labels": labels,
            }, ensure_ascii=False) + "\n")
            made += 1
            if made % 500 == 0:
                print(f"已生成 {made}/{neg_target}")

    # 校验: 重读一条, 检查长度与标签帧数一致, enrollment 存在
    with open(labels_path, encoding="utf-8") as f:
        rec = json.loads(f.readline())
    data, sr = sf.read(str(ROOT / rec["path"]))
    exp_frames = (len(data) + FRAME - 1) // FRAME
    ok = (sr == SR and len(rec["labels"]) == exp_frames
          and (ROOT / rec["enrollment"]).exists()
          and rec["enrollment"] not in rec["path"])
    print(f"校验 {rec['id']}: sr={sr}, 采样点={len(data)}, "
          f"标签帧数={len(rec['labels'])}, 期望={exp_frames}, "
          f"enrollment={rec['enrollment']}, {'OK' if ok else 'MISMATCH'}")
    print(f"完成: {labels_path} ({made} 条)")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
