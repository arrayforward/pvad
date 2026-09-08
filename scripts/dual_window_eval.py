# -*- coding: utf-8 -*-
"""双窗口交替（staggered dual-window）流式 PVAD 原型评估。

思路：两个流实例错开半个窗长跑（A 复位点 0,W,2W...，B 复位点 W/2,3W/2...），
单实例流长永不超过窗长 W（进不了 GRU 吸收态）；融合取"窗龄 > warmup 的成熟实例"，
都成熟时取年长者（rule=older）或两者较大值（rule=max）——冷启动区总被另一实例覆盖。

用法:
  python scripts/dual_window_eval.py --grid-val        # val 网格选型（W x warmup x rule）
  python scripts/dual_window_eval.py --test --W 8 --warmup 20 --rule older  # test 报数
  python scripts/dual_window_eval.py --script90 ...    # 90s 剧本
  python scripts/dual_window_eval.py --short ...       # 标准短句 100 条退化检查
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
CONFIRM = 4          # 与生产流式门控一致
HYST = 0.2
THR = 0.5
STREAM_ONNX = ROOT / "models/pvad/pvad_v4_stream.onnx"


def ema_feats(pcm, m0):
    raw = fbank(pcm).astype(np.float64)
    m = m0.copy()
    out = np.empty_like(raw)
    for i in range(len(raw)):
        m = 0.98 * m + 0.02 * raw[i]
        out[i] = raw[i] - m
    return out.astype(np.float32)


def dual_p2_fixed(sess, feats, emb, W_frames, warm_frames, rule):
    """正确版：保留 3 类 logits，per-frame softmax 后取目标类。"""
    T = len(feats)
    h = np.zeros((2, 2, 128), dtype=np.float32)
    age = [0, 0]
    out = np.zeros((2, T), dtype=np.float32)
    for s in range(0, T, CHUNK):
        for inst in (0, 1):
            off = inst * (W_frames // 2)
            if s > 0 and s >= off and (s - off) % W_frames == 0:
                h[:, inst] = 0.0
                age[inst] = 0
        lg, h = sess.run(None, {"feats_chunk": feats[None, s:s + CHUNK],
                                "emb": np.repeat(emb[None], 2, axis=0).astype(np.float32),
                                "h0": h})
        e = np.exp(lg - lg.max(-1, keepdims=True))
        e /= e.sum(-1, keepdims=True)
        cn = lg.shape[1]
        out[:, s:s + cn] = e[:, :, 2]  # [B=2, chunk]
        for inst in (0, 1):
            age[inst] += cn
    # 融合
    fused = np.zeros(T, dtype=np.float32)
    ageA = np.zeros(T)
    ageB = np.zeros(T)
    t_arr = np.arange(T)
    for inst, arr in ((0, ageA), (1, ageB)):
        off = inst * (W_frames // 2)
        arr[:] = (t_arr - off) % W_frames
        if off > 0:
            n0 = min(off, T)  # off 可能超过短文件长度
            arr[:n0] = t_arr[:n0]  # B 实例在 off 前窗龄=t（从流起始跑起）
    for t in range(T):
        cands = []
        if ageA[t] > warm_frames:
            cands.append((ageA[t], out[0, t]))
        if ageB[t] > warm_frames:
            cands.append((ageB[t], out[1, t]))
        if not cands:
            fused[t] = 0.0
        elif rule == "max":
            fused[t] = max(c[1] for c in cands)
        else:  # older
            fused[t] = max(cands)[1] if len(cands) > 1 else cands[0][1]
    return fused


def gate_events(p2, vad=None, warmup_vad=0, confirm=CONFIRM):
    consec = 0
    warm = warmup_vad
    ev = []
    for t, p in enumerate(p2):
        if vad is not None and vad[t] <= 0.5:
            continue
        if warm > 0:
            warm -= 1
            consec = 0
            continue
        if p > THR:
            consec += 1
            if consec >= confirm:
                ev.append(t)
                consec = 0
        elif p < THR - HYST:
            consec = 0
    return ev


def segments(labels):
    segs = []
    s = 0
    for t in range(1, len(labels) + 1):
        if t == len(labels) or labels[t] != labels[s]:
            segs.append((s, t - 1, int(labels[s])))
            s = t
    return segs


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
    n_f = len(pcm) // 160
    out = np.zeros(n_f)
    for fidx in range(n_f):
        out[fidx] = vp[min(len(vp) - 1, int(fidx * 160 / 512))]
    return out


class DualCfg:
    def __init__(self, W_s, warm, rule):
        self.W = int(W_s * 100)
        self.warm = warm
        self.rule = rule

    def __str__(self):
        return f"W={self.W / 100:.0f}s warm={self.warm} {self.rule}"


def dual_run(sess, feats, emb, cfg, vframe):
    """双窗口 + 生产门控（VAD 门控 + 全局 warm-up + confirm=4）。"""
    p2 = dual_p2_fixed(sess, feats, emb, cfg.W, cfg.warm, cfg.rule)
    return gate_events(p2, vad=vframe, warmup_vad=20)


def baseline_run(sess, feats, emb, vframe):
    """基线：单实例 + 会话策略（VAD speech-end 复位 + warmup20 + confirm=4）。"""
    T = len(feats)
    h = np.zeros((2, 1, 128), dtype=np.float32)
    out = np.zeros(T, dtype=np.float32)
    for s in range(0, T, CHUNK):
        # VAD speech-end 复位
        if s > 0 and s < len(vframe) and vframe[s - 1] > 0.5 and vframe[s] <= 0.5:
            h = np.zeros((2, 1, 128), dtype=np.float32)
        lg, h = sess.run(None, {"feats_chunk": feats[None, s:s + CHUNK],
                                "emb": emb[None].astype(np.float32), "h0": h})
        e = np.exp(lg - lg.max(-1, keepdims=True))
        e /= e.sum(-1, keepdims=True)
        out[s:s + lg.shape[1]] = e[0][:, 2]
    return gate_events(out, vad=vframe, warmup_vad=20)


def eval_long(td, sess, embder, cfg, max_n, baseline=False):
    recs = load_labels(td / "labels.jsonl")[:max_n]
    seg_tot = {2: 0, 1: 0, 0: 0}
    seg_hit = {2: 0, 1: 0, 0: 0}
    pn_streams = pn_clean = 0
    for r in recs:
        pcm, _ = read_wav(ROOT / r["path"])
        epcm, _ = read_wav(ROOT / r["enrollment"])
        m0 = fbank(epcm).astype(np.float64).mean(axis=0)
        emb = embder.embed(epcm)
        emb = emb / np.linalg.norm(emb)
        feats = ema_feats(pcm, m0)
        labels = np.asarray(r["labels"])
        T = min(len(feats), len(labels))
        feats = feats[:T]
        vframe = silero_probs(pcm)[:T]
        ev = baseline_run(sess, feats, emb, vframe) if baseline else dual_run(sess, feats, emb, cfg, vframe)
        if r.get("negative"):
            pn_streams += 1
            if not ev:
                pn_clean += 1
        for s, e, lab in segments(labels[:T]):
            if e - s < 10:
                continue
            seg_tot[lab] += 1
            if any(s <= f <= e for f in ev):
                seg_hit[lab] += 1
    n = len(recs)
    return {"tgt_rec": seg_hit[2] / max(seg_tot[2], 1),
            "nontgt_false": seg_hit[1] / max(seg_tot[1], 1),
            "noise_false": seg_hit[0] / max(seg_tot[0], 1),
            "pn_clean": pn_clean / max(pn_streams, 1),
            "n": n}


def eval_script90(sess, embder, cfg, baseline=False):
    from longstream_check import SCRIPT, LS
    pcm, _ = read_wav(LS / "script90.wav")
    epcm, _ = read_wav(LS / "enroll.wav")
    emb = embder.embed(epcm)
    emb = emb / np.linalg.norm(emb)
    m0 = fbank(epcm).astype(np.float64).mean(axis=0)
    feats = ema_feats(pcm, m0)
    vframe = silero_probs(pcm)
    ev = baseline_run(sess, feats, emb, vframe) if baseline else dual_run(sess, feats, emb, cfg, vframe)
    t0 = 0
    ok = noise_false = nontgt_false = 0
    for lab, dur, kind in SCRIPT:
        t1 = t0 + int(dur * 100)
        fires = [f for f in ev if t0 <= f < t1]
        if lab == 2:
            ok += bool(fires)
        elif lab == 0:
            noise_false += bool(fires)
        else:
            nontgt_false += bool(fires)
        t0 = t1
    return ok, noise_false, nontgt_false


def eval_short(sess, embder, cfg, max_n, baseline=False):
    """标准短句 e2e（data/mixtures/test 前 100 条，eval_pvad 协议：±20 帧容忍）。"""
    td = ROOT / "data" / "mixtures" / "test"
    recs = [r for r in load_labels(td / "labels.jsonl") if r.get("overlap_frames")][:max_n]
    ok = miss = false = 0
    for r in recs:
        pcm, _ = read_wav(ROOT / r["path"])
        epcm, _ = read_wav(ROOT / r["enrollment"])
        m0 = fbank(epcm).astype(np.float64).mean(axis=0)
        emb = embder.embed(epcm)
        emb = emb / np.linalg.norm(emb)
        feats = ema_feats(pcm, m0)
        vframe = silero_probs(pcm)
        ev = baseline_run(sess, feats, emb, vframe) if baseline else dual_run(sess, feats, emb, cfg, vframe)
        trig = ev[0] if ev else None
        TOL = 20
        if trig is None:
            miss += 1
            continue
        hit = any(st - TOL <= trig <= ed + TOL for st, ed in r["overlap_frames"])
        if not hit and r["labels"][min(trig, len(r["labels"]) - 1)] == 2:
            hit = True
        ok += hit
        false += not hit
    n = len(recs)
    return {"ok": ok / n, "miss": miss / n, "false": false / n, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-val", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--script90", action="store_true")
    ap.add_argument("--short", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--W", type=float, default=8)
    ap.add_argument("--W-list", type=float, nargs="*", default=None)  # 网格分片续跑用
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rule", choices=["older", "max"], default="older")
    ap.add_argument("--max-n", type=int, default=60)
    ap.add_argument("--max-n-test", type=int, default=300)
    args = ap.parse_args()

    sess = ort.InferenceSession(str(STREAM_ONNX), providers=["CPUExecutionProvider"])
    embder = CampplusEmbedder(intra_threads=4)
    cfg = DualCfg(args.W, args.warmup, args.rule)

    if args.grid_val:
        td = ROOT / "data/mixtures_long/val"
        W_list = args.W_list
        if not W_list:
            print("=== 基线（单实例+会话策略） val ===")
            b = eval_long(td, sess, embder, None, args.max_n, baseline=True)
            print(f"  baseline: tgt_rec={b['tgt_rec']:.3f} nontgt_false={b['nontgt_false']:.3f} "
                  f"pn_clean={b['pn_clean']:.3f}")
            b90 = eval_script90(sess, embder, None, baseline=True)
            print(f"  baseline script90: 目标 {b90[0]}/4 噪声 {b90[1]} 非目标 {b90[2]}")
            W_list = (6, 8, 12)
        print(f"=== 网格（val, W={W_list}）===")
        rows = []
        for W in W_list:
            for warm in (10, 20, 30):
                for rule in ("older", "max"):
                    c = DualCfg(W, warm, rule)
                    m = eval_long(td, sess, embder, c, args.max_n)
                    s90 = eval_script90(sess, embder, c)
                    rows.append((c, m, s90))
                    print(f"  {c}: tgt_rec={m['tgt_rec']:.3f} nontgt_false={m['nontgt_false']:.3f} "
                          f"pn={m['pn_clean']:.3f} | 90s: tgt {s90[0]}/4 noise {s90[1]} nontgt {s90[2]}",
                          flush=True)
        print("=== 网格汇总（按 90s 非目标误触发升序，然后 val 召回降序）===")
        rows.sort(key=lambda x: (x[2][2] + x[2][1], -x[1]["tgt_rec"]))
        for c, m, s90 in rows:
            print(f"  {c}: 90s(tgt{s90[0]}/4 noise{s90[1]} nontgt{s90[2]}) "
                  f"val(rec={m['tgt_rec']:.3f} false={m['nontgt_false']:.3f} pn={m['pn_clean']:.3f})")
        return 0

    if args.script90:
        b = "基线" if args.baseline else f"双窗 {cfg}"
        ok, nf, nt = eval_script90(sess, embder, cfg, baseline=args.baseline)
        print(f"=== 90s 剧本（{b}）: 目标 {ok}/4 噪声 {nf} 非目标 {nt} ===")
    if args.test:
        td = ROOT / "data/mixtures_long/test"
        b = "基线" if args.baseline else f"双窗 {cfg}"
        m = eval_long(td, sess, embder, cfg, args.max_n_test, baseline=args.baseline)
        print(f"=== 长流 test n={m['n']}（{b}）: tgt_rec={m['tgt_rec']:.3f} "
              f"nontgt_false={m['nontgt_false']:.3f} noise_false={m['noise_false']:.3f} "
              f"pn_clean={m['pn_clean']:.3f} ===")
    if args.short:
        b = "基线" if args.baseline else f"双窗 {cfg}"
        m = eval_short(sess, embder, cfg, 100, baseline=args.baseline)
        print(f"=== 短句 100 条（{b}）: ok={m['ok']:.3f} miss={m['miss']:.3f} false={m['false']:.3f} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
