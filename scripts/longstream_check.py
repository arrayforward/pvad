# -*- coding: utf-8 -*-
"""长流（90s 剧本）流式 PVAD 诊断与修复策略验证脚本。

剧本（test_audio/longstream/script90.wav + script90_labels.json）：
  噪声10s → 目标5s → 旁人10s → 回声15s → 目标5s → 噪声15s → 目标5s → 旁人10s → 噪声15s → 目标5s

用法:
  python scripts/longstream_check.py --build   # 重建剧本（需要 data/speakers + MUSAN）
  python scripts/longstream_check.py           # 跑诊断：原始/复位/VAD门控策略对比
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvad_common import ROOT, fbank, read_wav, CampplusEmbedder  # noqa: E402

LS = ROOT / "test_audio" / "longstream"


def read16(p):
    w = wave.open(str(p), "rb")
    return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


SCRIPT = [  # (label, dur_s, kind)  kind: n=noise a=target b=other c=echo
    (0, 10, "n"), (2, 5, "a"), (1, 10, "b"), (1, 15, "c"), (2, 5, "a"),
    (0, 15, "n"), (2, 5, "a"), (1, 10, "b"), (0, 15, "n"), (2, 5, "a"),
]


def build():
    A = sorted(glob.glob(str(ROOT / "data/speakers/stcmds/P00190/*.wav")))
    B = sorted(glob.glob(str(ROOT / "data/speakers/stcmds/P00089/*.wav")))
    C = sorted(glob.glob(str(ROOT / "data/speakers/stcmds/P00351/*.wav")))
    enroll = [read16(A[0]), read16(A[1])]
    a_pool = [read16(f) for f in A[2:40]]
    b_pool = [read16(f) for f in B[2:20]]
    c_pool = [read16(f) for f in C[2:20]]
    noise = read16(ROOT / "data/raw/aug/musan/noise/free-sound/noise-free-sound-0000.wav") * 0.3
    sa, sb, sc = [2], [0], [0]

    def take(pool, dur, st):
        need = int(dur * 16000)
        out = []
        while need > 0:
            x = pool[st[0] % len(pool)]
            st[0] += 1
            k = min(len(x), need)
            out.append(x[:k])
            need -= k
        return np.concatenate(out)

    def nz(dur):
        need = int(dur * 16000)
        return np.tile(noise, int(np.ceil(need / len(noise))))[:need]

    parts, labels = [], []
    for lab, dur, kind in SCRIPT:
        seg = nz(dur) if kind == "n" else take({"a": a_pool, "b": b_pool, "c": c_pool}[kind], dur,
                                               {"a": sa, "b": sb, "c": sc}[kind])
        parts.append(seg)
        labels.extend([lab] * int(dur * 100))
    pcm = np.concatenate(parts)
    LS.mkdir(parents=True, exist_ok=True)

    def write_wav(path, x):
        w = wave.open(str(path), "wb")
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
        w.close()

    write_wav(LS / "script90.wav", pcm)
    write_wav(LS / "enroll.wav", np.concatenate(enroll))
    with open(LS / "script90_labels.json", "w", encoding="utf-8") as f:
        json.dump({"frame_ms": 10,
                   "script": [{"label": l, "dur": d, "kind": k} for l, d, k in SCRIPT],
                   "labels": labels}, f)
    print(f"built {LS}/script90.wav ({len(pcm) / 16000:.1f}s) + labels + enroll.wav")


def silero_probs(pcm):
    vs = ort.InferenceSession(str(ROOT / "models/silero_vad.onnx"),
                              providers=["CPUExecutionProvider"])
    state = np.zeros((2, 1, 128), dtype=np.float32)
    ctx = np.zeros((1, 64), dtype=np.float32)
    vp = []
    for i in range(0, len(pcm) - 511, 512):
        x = np.concatenate([ctx, pcm[i:i + 512][None, :].astype(np.float32)], axis=1)
        o, state = vs.run(None, {"input": x, "state": state,
                                 "sr": np.array(16000, dtype=np.int64)})
        ctx = x[:, -64:]
        vp.append(float(o[0, 0]))
    # 对齐到 10ms 帧（粗对齐：每 512 采样块覆盖 3.2 帧）
    n_f = len(pcm) // 160
    out = np.zeros(n_f)
    for fidx in range(n_f):
        out[fidx] = vp[min(len(vp) - 1, int(fidx * 160 / 512))]
    return out


def stream_p2(feats, emb, bounds=()):
    sess = ort.InferenceSession(str(ROOT / "models/pvad/pvad_v4_stream.onnx"),
                                providers=["CPUExecutionProvider"])
    h = np.zeros((2, 1, 128), dtype=np.float32)
    outs = []
    for s in range(0, len(feats), 5):
        if s in bounds and s > 0:
            h = np.zeros((2, 1, 128), dtype=np.float32)
        lg, h = sess.run(None, {"feats_chunk": feats[None, s:s + 5].astype(np.float32),
                                "emb": emb[None].astype(np.float32), "h0": h})
        outs.append(lg[0])
    lg = np.concatenate(outs, 0)
    e = np.exp(lg - lg.max(-1, keepdims=True))
    e /= e.sum(-1, keepdims=True)
    return e[:, 2]


_CONFIRM = 2


def gate_scan(p2, vframe, warmup_vad_frames, label_ranges):
    """策略：VAD>0.5 才更新门控；warmup_vad_frames 个 VAD 帧内不门控（每个新会话起始）。"""
    consec = 0
    warm = warmup_vad_frames
    events = []  # (frame, kind) kind: fire
    for t in range(len(p2)):
        if vframe[t] > 0.5:
            if warm > 0:
                warm -= 1
                consec = 0
                continue
            if p2[t] > 0.5:
                consec += 1
                if consec >= _CONFIRM:
                    events.append(t)
                    consec = 0  # 边沿触发后重新计（每段只报首个）
            elif p2[t] < 0.3:
                consec = 0
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()
    if args.build:
        build()
        return 0

    pcm, _ = read_wav(LS / "script90.wav")
    epcm, _ = read_wav(LS / "enroll.wav")
    labels = json.load(open(LS / "script90_labels.json", encoding="utf-8"))
    script = labels["script"]
    embder = CampplusEmbedder()
    emb = embder.embed(epcm)
    emb = emb / np.linalg.norm(emb)
    m0 = fbank(epcm).astype(np.float64).mean(axis=0)
    raw = fbank(pcm).astype(np.float64)
    m = m0.copy()
    feats = []
    for i in range(len(raw)):
        m = 0.98 * m + 0.02 * raw[i]
        feats.append((raw[i] - m).astype(np.float32))
    feats = np.stack(feats)
    vframe = silero_probs(pcm)

    bounds = set()
    t = 0
    for seg in script:
        t += int(seg["dur"] * 100)
        bounds.add(t)

    p_plain = stream_p2(feats, emb)
    p_reset = stream_p2(feats, emb, bounds)

    def report(p2, vg, tag, warmup):
        ev = gate_scan(p2, vg, warmup, None)
        print(f"--- {tag} (warmup={warmup} VAD帧) ---")
        t0 = 0
        ok = miss = false = 0
        for seg in script:
            t1 = t0 + int(seg["dur"] * 100)
            fires = [e for e in ev if t0 <= e < t1]
            mark = {0: "noise", 1: "nontgt", 2: "target"}[seg["label"]]
            if seg["label"] == 2:
                v = "OK" if fires else "MISS"
                ok += bool(fires)
                miss += not fires
            else:
                v = "FALSE" if fires else "ok"
                false += bool(fires)
            print(f"  t={t0 / 100:5.1f}-{t1 / 100:5.1f} {mark:6s} meanP={p2[t0:t1].mean():.3f} "
                  f"fire@{fires[0] if fires else '-'} {v}")
            t0 = t1
        print(f"  => target_ok={ok}/4 miss={miss} false={false}")

    print("=== silero VAD 分段 frac>0.5 ===")
    t0 = 0
    for seg in script:
        t1 = t0 + int(seg["dur"] * 100)
        frac = float(np.mean(vframe[t0:t1] > 0.5))
        mark = {0: "noise", 1: "nontgt", 2: "target"}[seg["label"]]
        print(f"  t={t0 / 100:5.1f}-{t1 / 100:5.1f} {mark:6s} frac={frac:.2f}")
        t0 = t1

    report(p_plain, vframe, "原始整流（无复位）", 0)
    report(p_reset, vframe, "段末复位 + VAD门控打分", args.warmup)

    # 周期复位：每 8s（=训练片段长度上限）强制新会话，从根上避免长时状态出分布
    p_period = stream_p2(feats, emb, set(range(800, len(feats), 800)))
    report(p_period, vframe, "每 8s 周期复位 + VAD门控打分", args.warmup)

    # confirm 加严：复位后 blip 多为孤立尖峰，目标段为持续高 P
    for cf in (3, 4, 5, 6):
        global _CONFIRM
        _CONFIRM = cf
        report(p_reset, vframe, f"段末复位 + confirm={cf}", args.warmup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
