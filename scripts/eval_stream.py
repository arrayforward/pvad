# -*- coding: utf-8 -*-
"""流式 PVAD 的 CMVN 方案评估与端到端确认。

整段 per-bin 均值归一化需要未来信息, 流式只能用因果近似:
  running  — 累积运行均值 (从流起始到当前帧)
  sliding  — 滑动窗均值 (最近 3s)
  ema      — 指数滑动均值 (alpha=0.02)

用法:
  python scripts/eval_stream.py --analyze --max-n 100   # 三方案对比 + warm-up
  python scripts/eval_stream.py --e2e --cmvn ema --max-n 200  # 端到端 vs 整段基线
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvad_common import ROOT, fbank, read_wav, load_labels  # noqa: E402

FULL_ONNX = ROOT / "models" / "pvad" / "pvad_v4.onnx"
STREAM_ONNX = ROOT / "models" / "pvad" / "pvad_v4_stream.onnx"
CHUNK = 5
FRAME_S = 0.01
USE_TOKENS = False  # --full-onnx/--stream-onnx 为 v5 时置 True


def cmvn_full(x):
    return x - x.mean(axis=0, keepdims=True)


def cmvn_running(x):
    return x - np.cumsum(x, axis=0) / np.arange(1, len(x) + 1)[:, None]


def cmvn_sliding(x, win=300):
    c = np.cumsum(x, axis=0)
    out = np.empty_like(x)
    for t in range(len(x)):
        s = max(0, t - win + 1)
        m = (c[t] - (c[s - 1] if s > 0 else 0)) / (t - s + 1)
        out[t] = x[t] - m
    return out


def cmvn_ema(x, alpha=0.02, m0=None):
    out = np.empty_like(x)
    m = np.zeros(x.shape[1], dtype=np.float64) if m0 is None else m0.copy()
    for t in range(len(x)):
        m = (1 - alpha) * m + alpha * x[t]
        out[t] = x[t] - m
    return out


def cmvn_running_prior(x, m0, n0=100):
    """运行均值, 用 prior 均值 m0 和伪计数 n0 初始化 (如 enrollment 的 fbank 均值)。"""
    c = np.cumsum(x, axis=0)
    n = np.arange(1, len(x) + 1)[:, None]
    return x - (n0 * m0[None, :] + c) / (n0 + n)


def cmvn_perframe(x, with_std=False):
    """per-frame CMVN: 每帧独立减均值 (v6; 无跨帧统计, 流式/离线天然一致)。"""
    y = x - x.mean(axis=1, keepdims=True)
    if with_std:
        y = y / (x.std(axis=1, keepdims=True) + 1e-5)
    return y


CMVN = {"running": cmvn_running, "sliding": cmvn_sliding, "ema": cmvn_ema,
        "perframe": cmvn_perframe}


def p2_full(sess, feats_cmvn, emb, tokens=None):
    if tokens is not None:
        lg = sess.run(None, {"feats": feats_cmvn[None].astype(np.float32),
                             "enroll_tokens": tokens[None].astype(np.float32),
                             "enroll_mask": np.zeros((1, len(tokens)), dtype=bool)})[0]
    else:
        lg = sess.run(None, {"feats": feats_cmvn[None].astype(np.float32),
                             "emb": emb[None]})[0]
    return softmax(lg[0])[:, 2]


def p2_stream(sess, feats_cmvn, emb, chunk=CHUNK, tokens=None):
    h = np.zeros((2, 1, 128), dtype=np.float32)
    outs = []
    v5 = tokens is not None
    for s in range(0, len(feats_cmvn), chunk):
        if v5:
            lg, h = sess.run(None, {
                "feats_chunk": feats_cmvn[None, s:s + chunk].astype(np.float32),
                "enroll_tokens": tokens[None].astype(np.float32),
                "enroll_mask": np.zeros((1, len(tokens)), dtype=bool), "h0": h})
        else:
            lg, h = sess.run(None, {
                "feats_chunk": feats_cmvn[None, s:s + chunk].astype(np.float32),
                "emb": emb[None], "h0": h})
        outs.append(lg[0])
    return softmax(np.concatenate(outs, axis=0))[:, 2]


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def gate_trigger(p2, thr=0.5, hyst=0.2, confirm=2):
    consec = 0
    for t, p in enumerate(p2):
        if p > thr:
            consec += 1
            if consec >= confirm:
                return t
        elif p < thr - hyst:
            consec = 0
    return None


def judge(trig, overlaps, labels):
    TOL = 20
    if trig is None:
        return "miss"
    for st, ed in overlaps:
        if st - TOL <= trig <= ed + TOL:
            return "ok"
    return "ok" if labels[min(trig, len(labels) - 1)] == 2 else "false"


def warmup_time(diffs, thr=0.05):
    """|diff| 首次降到 thr 以下且之后不再超过的时刻 (秒); 不超过则返回 None。"""
    above = np.where(diffs >= thr)[0]
    if len(above) == 0:
        return 0.0
    last = above[-1]
    if last == len(diffs) - 1:
        return None
    return (last + 1) * FRAME_S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--e2e", action="store_true")
    ap.add_argument("--cmvn", choices=["running", "sliding", "ema",
                                       "running_prior", "ema_prior", "perframe"],
                    default="ema")
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--prior-n0", type=int, default=100,
                    help="running_prior 的伪计数 (enrollment 均值先验强度)")
    ap.add_argument("--max-n", type=int, default=100)
    ap.add_argument("--full-onnx", default=None)
    ap.add_argument("--stream-onnx", default=None)
    args = ap.parse_args()

    full_path = args.full_onnx or str(FULL_ONNX)
    stream_path = args.stream_onnx or str(STREAM_ONNX)
    sess_full = ort.InferenceSession(full_path, providers=["CPUExecutionProvider"])
    sess_stream = ort.InferenceSession(stream_path, providers=["CPUExecutionProvider"])
    # v5/v6 模型的流式输入含 enroll_tokens
    use_tokens = "enroll_tokens" in [i.name for i in sess_stream.get_inputs()]

    for tag, td in (("干净", ROOT / "data" / "mixtures" / "test"),
                    ("增广", ROOT / "data" / "mixtures_v2" / "test")):
        recs = [r for r in load_labels(td / "labels.jsonl")
                if r.get("overlap_frames")][: args.max_n]
        base_stats, var_stats = {}, {}
        per_variant = {k: {"maxdiff": [], "shift": [], "warmup": [],
                           "gate": {"ok": 0, "miss": 0, "false": 0}}
                       for k in ["running", "sliding", "ema",
                                 "running_prior", "ema_prior", "perframe"]}
        base_gate = {"ok": 0, "miss": 0, "false": 0}
        for r in recs:
            pcm, _ = read_wav(ROOT / r["path"])
            raw = fbank(pcm).astype(np.float64)
            if use_tokens:
                emb = None
                toks = np.load(td / "emb_tokens" / f"{r['id']}.npy")
            else:
                emb = np.load(td / "emb" / f"{r['id']}.npy")
                toks = None
            labels = np.asarray(r["labels"])
            T = min(len(raw), len(labels))
            raw, labels = raw[:T], labels[:T]
            # enrollment fbank 均值先验 (同说话人同信道)
            epcm, _ = read_wav(ROOT / r["enrollment"])
            m0 = fbank(epcm).astype(np.float64).mean(axis=0)
            # v6 (perframe): 基线用同一逐帧归一化 (离线=流式同分布)
            base_fx = (cmvn_perframe(raw) if args.cmvn == "perframe"
                       else cmvn_full(raw))
            p_base = p2_full(sess_full, base_fx.astype(np.float32), emb,
                             tokens=toks)
            trig_base = gate_trigger(p_base)
            if args.e2e:
                base_gate[judge(trig_base, r["overlap_frames"], labels)] += 1
                variants = [args.cmvn]
            else:
                variants = ["running", "sliding", "ema"]
            for name in variants:
                if name == "running_prior":
                    fx = cmvn_running_prior(raw, m0, args.prior_n0)
                elif name == "ema_prior":
                    fx = cmvn_ema(raw, args.alpha, m0)
                elif name == "ema":
                    fx = cmvn_ema(raw, args.alpha)
                else:
                    fx = CMVN[name](raw)
                pv = p2_stream(sess_stream, fx.astype(np.float32), emb, tokens=toks)
                d = np.abs(p_base - pv)
                st = per_variant[name]
                st["maxdiff"].append(d.max())
                tb, tv = gate_trigger(p_base), gate_trigger(pv)
                st["shift"].append(abs((tb or -1) - (tv or -1)))
                st["warmup"].append(warmup_time(d))
                if args.e2e:
                    st["gate"][judge(tv, r["overlap_frames"], labels)] += 1

        if args.analyze:
            print(f"\n=== [{tag}] {len(recs)} 条, running CMVN 近似影响 ===")
            print(f"{'方案':<9}{'max|P差| 中位/95%':>20}{'触发帧偏移 中位/95%':>20}"
                  f"{'warm-up 中位/95%':>18}")
            for name, st in per_variant.items():
                md = np.percentile(st["maxdiff"], [50, 95])
                sh = np.percentile(st["shift"], [50, 95])
                wu = [w for w in st["warmup"] if w is not None]
                wu_s = (f"{np.percentile(wu, 50):.2f}s/{np.percentile(wu, 95):.2f}s"
                        if wu else "n/a")
                print(f"{name:<9}{md[0]:>10.4f}/{md[1]:<9.4f}"
                      f"{sh[0]:>10.0f}/{sh[1]:<9.0f}{wu_s:>18}")
        if args.e2e:
            n = len(recs)
            st = per_variant[args.cmvn]
            print(f"\n=== [{tag}] 流式 e2e ({args.cmvn} CMVN, chunk={CHUNK}, n={n}) ===")
            for name, g in (("整段基线", base_gate), (f"流式-{args.cmvn}", st["gate"])):
                tot = g["ok"] + g["miss"] + g["false"]
                print(f"{name:<14} 漏 {g['miss'] / tot:.3f} "
                      f"误 {g['false'] / tot:.3f} 正确 {g['ok'] / tot:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
