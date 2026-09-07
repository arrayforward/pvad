# -*- coding: utf-8 -*-
"""长流评估/验收: 流式 PVAD (state 外置 ONNX) 在长流数据上的门控指标。

两部分:
  A. data/mixtures_long/<split>: 逐条流式推理 (EMA α=0.02, chunk=5, 零初始态),
     confirm=2 门控, 按段统计:
       - 目标段召回: 每个连续 label==2 段内是否有触发 (首个触发帧即算)
       - 非目标误触发: label!=2 段上的触发次数 (纯负样本流应全程 0)
  B. test_audio/longstream/script90.wav 90s 剧本 (longstream_check 同协议:
     silero VAD 门控 + warmup 20 + 段末复位, confirm=2):
     目标段 4/4、噪声段 ≤1、非目标语音段 ≤1
  C. --cold-start: 冷启动前 50 帧 P(target) 曲线 (非目标语音流上, 与参照模型对比)

用法:
  python scripts/eval_longstream.py --stream-onnx models/pvad/pvad_v7_stream.onnx \
      --split val --max-n 150
  python scripts/eval_longstream.py --stream-onnx ... --script90
  python scripts/eval_longstream.py --stream-onnx ... --cold-start [--ref-onnx models/pvad/pvad_v4_stream.onnx]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvad_common import ROOT, fbank, read_wav, load_labels, CampplusEmbedder  # noqa: E402

CHUNK = 5
CONFIRM = 2
HYST = 0.2
THR = 0.5


def ema_feats(pcm, m0):
    raw = fbank(pcm).astype(np.float64)
    m = m0.copy()
    out = np.empty_like(raw)
    for i in range(len(raw)):
        m = 0.98 * m + 0.02 * raw[i]
        out[i] = raw[i] - m
    return out.astype(np.float32)


def stream_p2(sess, feats, emb, bounds=()):
    h = np.zeros((2, 1, 128), dtype=np.float32)
    outs = []
    for s in range(0, len(feats), CHUNK):
        if s in bounds and s > 0:
            h = np.zeros((2, 1, 128), dtype=np.float32)
        lg, h = sess.run(None, {"feats_chunk": feats[None, s:s + CHUNK],
                                "emb": emb[None].astype(np.float32), "h0": h})
        outs.append(lg[0])
    lg = np.concatenate(outs, 0)
    e = np.exp(lg - lg.max(-1, keepdims=True))
    e /= e.sum(-1, keepdims=True)
    return e[:, 2]


def gate_events(p2, thr=THR, hyst=HYST, confirm=CONFIRM, vad=None, warmup=0):
    consec = 0
    warm = warmup
    ev = []
    for t, p in enumerate(p2):
        if vad is not None and vad[t] <= 0.5:
            continue
        if warm > 0:
            warm -= 1
            consec = 0
            continue
        if p > thr:
            consec += 1
            if consec >= confirm:
                ev.append(t)
                consec = 0
        elif p < thr - hyst:
            consec = 0
    return ev


def segments(labels):
    """连续同标签段 [(s, e, lab)]。"""
    segs = []
    s = 0
    for t in range(1, len(labels) + 1):
        if t == len(labels) or labels[t] != labels[s]:
            segs.append((s, t - 1, int(labels[s])))
            s = t
    return segs


def eval_split(sess, td, embder, max_n):
    recs = load_labels(td / "labels.jsonl")[:max_n]
    seg_tot = {2: 0, 1: 0, 0: 0}
    seg_hit = {2: 0, 1: 0, 0: 0}
    pure_neg_streams = 0
    pure_neg_clean = 0
    n = 0
    for r in recs:
        feats = np.load(td / "feats_ema" / f"{r['id']}.npy")
        emb = np.load(td / "emb" / f"{r['id']}.npy")
        labels = np.asarray(r["labels"])
        T = min(len(feats), len(labels))
        p2 = stream_p2(sess, feats[:T], emb)
        ev = gate_events(p2)
        if r.get("negative"):
            pure_neg_streams += 1
            if not ev:
                pure_neg_clean += 1
        for s, e, lab in segments(labels[:T]):
            if e - s < 10:  # 忽略 <0.1s 碎片
                continue
            seg_tot[lab] += 1
            if any(s <= f <= e for f in ev):
                seg_hit[lab] += 1
        n += 1
    print(f"  目标段召回 {seg_hit[2]}/{seg_tot[2]} "
          f"非目标误触发段 {seg_hit[1]}/{seg_tot[1]} "
          f"噪声误触发段 {seg_hit[0]}/{seg_tot[0]}")
    print(f"  纯负样本流 {pure_neg_streams} 条, 全程零触发 {pure_neg_clean} 条")
    return {"tgt_rec": seg_hit[2] / max(seg_tot[2], 1),
            "nontgt_false": seg_hit[1] / max(seg_tot[1], 1),
            "pure_neg_clean": pure_neg_clean / max(pure_neg_streams, 1)}


def eval_script90(sess, embder, warmup=20):
    from longstream_check import silero_probs, SCRIPT, LS
    pcm, _ = read_wav(LS / "script90.wav")
    epcm, _ = read_wav(LS / "enroll.wav")
    emb = embder.embed(epcm)
    m0 = fbank(epcm).astype(np.float64).mean(axis=0)
    feats = ema_feats(pcm, m0)
    vframe = silero_probs(pcm)
    bounds = set()
    t = 0
    for seg in SCRIPT:
        t += int(seg[1] * 100)
        bounds.add(t)
    p2 = stream_p2(sess, feats, emb, bounds)
    ev = gate_events(p2, vad=vframe, warmup=warmup)
    t0 = 0
    ok = noise_false = nontgt_false = 0
    lines = []
    for lab, dur, kind in SCRIPT:
        t1 = t0 + int(dur * 100)
        fires = [f for f in ev if t0 <= f < t1]
        mark = {"n": "noise", "a": "target", "b": "nontgt", "c": "echo"}[kind]
        if lab == 2:
            ok += bool(fires)
        elif lab == 0:
            noise_false += bool(fires)
        else:
            nontgt_false += bool(fires)
        lines.append(f"  t={t0 / 100:5.1f}-{t1 / 100:5.1f} {mark:6s} "
                     f"meanP={p2[t0:t1].mean():.3f} fires={len(fires)}")
        t0 = t1
    for l in lines:
        print(l)
    print(f"  => 目标段 {ok}/4, 噪声误触发 {noise_false}, 非目标语音误触发 {nontgt_false}")
    return ok, noise_false, nontgt_false


def cold_start(sess, ref_sess, td, embder):
    """非目标长语音流上, 零初始态前 50 帧 P(target) 对比。"""
    recs = [r for r in load_labels(td / "labels.jsonl")
            if r.get("negative") and r["duration_s"] > 20][:3]
    for r in recs[:1]:
        feats = np.load(td / "feats_ema" / f"{r['id']}.npy")
        emb = np.load(td / "emb" / f"{r['id']}.npy")
        for name, s in (("new", sess), ("ref_v4s", ref_sess)):
            if s is None:
                continue
            p2 = stream_p2(s, feats[:60], emb)
            pts = " ".join(f"{p:.2f}" for p in p2[:50:5])
            print(f"  {name:8s} P(target) f0-f45 (每5帧): {pts} | "
                  f"mean={p2[:50].mean():.3f} max={p2[:50].max():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream-onnx", required=True)
    ap.add_argument("--ref-onnx", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--max-n", type=int, default=150)
    ap.add_argument("--script90", action="store_true")
    ap.add_argument("--cold-start", action="store_true")
    args = ap.parse_args()

    sess = ort.InferenceSession(args.stream_onnx, providers=["CPUExecutionProvider"])
    ref = ort.InferenceSession(args.ref_onnx, providers=["CPUExecutionProvider"]) \
        if args.ref_onnx else None
    embder = CampplusEmbedder(intra_threads=4)

    if args.split:
        td = ROOT / "data" / "mixtures_long" / args.split
        print(f"=== 长流 {args.split} (n≤{args.max_n}) {Path(args.stream_onnx).name} ===")
        eval_split(sess, td, embder, args.max_n)
    if args.script90:
        print(f"=== 90s 剧本 ({Path(args.stream_onnx).name}) ===")
        eval_script90(sess, embder)
    if args.cold_start:
        td = ROOT / "data" / "mixtures_long" / (args.split or "test")
        print(f"=== 冷启动 P 曲线 (负样本流, {td.name}) ===")
        cold_start(sess, ref, td, embder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
