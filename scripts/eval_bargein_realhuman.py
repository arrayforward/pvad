# -*- coding: utf-8 -*-
"""barge-in 对照实验：当前架构（v7b 帧级触发） vs 去掉 v7b（v4 周期判决承担打断）。

真人 test 池场景：B（干扰）从 0s 起播放，A（注册用户）在 t0∈[1,3]s 开口
（A 有效 RMS 0.03~0.06，B 低 5~15dB），注册 A 的其它 3-10s 句子。

被测打断路径（同批样本配对比较）：
  v7b        : 10ms 帧级流式，P>0.5 confirm=2 hyst=0.2（当前生产）
  v4-3s1s    : direct_v4 官方点 TH=0.23/UNC=0.2，因果 3s 滑窗 / 1s hop（头部零填充）
  v4-1.5s0.5s: 同上但 1.5s 窗 / 0.5s hop（降延迟变体，偏离其 3s 训练窗）

判定：首次触发/open 时刻 t：< t0-0.2s 为 B 期间误触发；>= t0-0.2s 为正确，
延迟 = t - t0；无触发为漏。B_only 样本任何触发均为误触发。

用法: python scripts/eval_bargein_realhuman.py --n 200 --neg 100 \
        --out build/eval_bargein_realhuman.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

DV = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DV / "scripts"))
from pvad_common import DATA_ROOT, fbank, read_wav  # noqa: E402
from gen_mixtures import build_track, load_manifest, rms  # noqa: E402
from eval_realhuman import Gates, active_rms  # noqa: E402
from compare_tse_pvad import gate_trigger  # noqa: E402

SR = 16000


def gen_bargein(rng, by_spk, speakers, with_a=True):
    """B 从 0s 起；A 在 t0 开口（with_a=False 时纯 B）。返回 dict 或 None。"""
    tgt_spk, itf_spk = rng.sample(speakers, 2)
    length = int(rng.uniform(5.0, 8.0) * SR)
    itf, mask_g, _ = build_track(rng, by_spk[itf_spk], length)
    if not (mask_g == 1).any():
        return None
    cands = [u for u in by_spk[tgt_spk] if 3.0 <= u[1] <= 10.0]
    if not cands:
        return None
    enroll = rng.choice(cands)[0]
    level = float(np.clip(rng.gauss(0.045, 0.008), 0.03, 0.06))
    snr_db = rng.uniform(5.0, 15.0)
    b_r = active_rms(itf, mask_g)
    gb = level / 10 ** (snr_db / 20.0) / b_r
    if not with_a:
        mix = np.clip(itf * gb, -0.98, 0.98).astype(np.float32)
        return {"enroll": enroll, "mix": mix, "t0": None, "snr_db": round(snr_db, 2)}
    t0 = rng.uniform(1.0, 3.0)
    off = int(t0 * SR)
    pool_t = [u for u in by_spk[tgt_spk] if u[0] != enroll]
    if not pool_t:
        return None
    tgt_tail, mask_t, _ = build_track(rng, pool_t, length - off)
    if not (mask_t == 1).any():
        return None
    tgt = np.zeros(length, dtype=np.float32)
    tgt[off:] = tgt_tail
    mask = np.zeros(length, dtype=np.int8)
    mask[off:] = mask_t
    ga = level / active_rms(tgt, mask)
    mix = np.clip(tgt * ga + itf * gb, -0.98, 0.98).astype(np.float32)
    return {"enroll": enroll, "mix": mix, "t0": t0, "snr_db": round(snr_db, 2)}


class V4Causal:
    """direct_v4 因果滑窗判决：t 时刻取过去 win 秒（不足左补零）。"""

    def __init__(self, gates, win_s, hop_s):
        self.gates = gates
        self.w = int(win_s * SR)
        self.hop = hop_s

    @torch.inference_mode()
    def first_open(self, wave, ref):
        """返回首个放行时刻（秒）；无放行返回 None。"""
        gate = self.gates.v4
        t = self.hop
        while t * SR <= len(wave):
            n = int(t * SR)
            seg = np.zeros(self.w, dtype=np.float32)
            tail = wave[max(0, n - self.w):n]
            seg[self.w - len(tail):] = tail
            emb = self.gates.enc(torch.from_numpy(seg)[None])
            _, _, open_ = self.gates.direct(gate, emb, ref)
            if open_:
                return t
            t += self.hop
        return None


def judge(t_trig, t0):
    """t0=None 表示纯 B 样本。"""
    if t0 is None:
        return "false" if t_trig is not None else "ok_closed"
    if t_trig is None:
        return "miss"
    return "false_pre" if t_trig < t0 - 0.2 else "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="A barge-in 样本数")
    ap.add_argument("--neg", type=int, default=100, help="B_only 样本数")
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--max-n", type=int, default=0)
    ap.add_argument("--th4", type=float, default=None)
    ap.add_argument("--unc4", type=float, default=None)
    ap.add_argument("--out", default=str(DV / "build" / "eval_bargein_realhuman.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_spk = load_manifest(DATA_ROOT / "manifest.jsonl")
    with open(DATA_ROOT / "split.json", encoding="utf-8") as f:
        pools = json.load(f)
    speakers = [s for s in pools["test"] if s in by_spk and len(by_spk[s]) >= 3]
    print(f"test 池真人说话人 {len(speakers)}")

    gates = Gates(th4=args.th4, unc4=args.unc4)  # 默认 v4 官方 TH=0.23/UNC=0.2
    th_desc = f"v4 TH={gates.v4.th} UNC={gates.v4.unc}"
    variants = {"v4-3s1s": V4Causal(gates, 3.0, 1.0),
                "v4-1.5s0.5s": V4Causal(gates, 1.5, 0.5)}
    stats = {}

    def rec(model, verdict, delay=None):
        st = stats.setdefault(model, {"ok": 0, "miss": 0, "false_pre": 0,
                                      "false": 0, "ok_closed": 0, "delays": []})
        st[verdict] += 1
        if delay is not None:
            st["delays"].append(delay)

    n_pos = args.max_n or args.n
    n_neg = max(2, args.max_n // 2) if args.max_n else args.neg
    t_start = time.time()
    made = 0
    while made < n_pos:
        s = gen_bargein(rng, by_spk, speakers, with_a=True)
        if s is None:
            continue
        made += 1
        enroll, _ = read_wav(s["enroll"])
        with torch.inference_mode():
            ref = gates.enc(torch.from_numpy(enroll.astype(np.float32))[None])[0].numpy()
        cam_emb = gates.cam.embed(enroll.astype(np.float32))
        m0 = fbank(enroll.astype(np.float32)).astype(np.float64).mean(axis=0)

        p2, trig, _ = gates.v7b(s["mix"], cam_emb, m0)
        v = judge(trig * 0.01 if trig is not None else None, s["t0"])
        rec("v7b", v, (trig * 0.01 - s["t0"]) if v == "ok" else None)
        for name, va in variants.items():
            t_open = va.first_open(s["mix"], ref)
            v = judge(t_open, s["t0"])
            rec(name, v, (t_open - s["t0"]) if v == "ok" else None)
        if made % 20 == 0:
            print(f"[A {made}/{n_pos}] elapsed={time.time()-t_start:.0f}s", flush=True)

    made = 0
    while made < n_neg:
        s = gen_bargein(rng, by_spk, speakers, with_a=False)
        if s is None:
            continue
        made += 1
        enroll, _ = read_wav(s["enroll"])
        with torch.inference_mode():
            ref = gates.enc(torch.from_numpy(enroll.astype(np.float32))[None])[0].numpy()
        cam_emb = gates.cam.embed(enroll.astype(np.float32))
        m0 = fbank(enroll.astype(np.float32)).astype(np.float64).mean(axis=0)
        _, trig, _ = gates.v7b(s["mix"], cam_emb, m0)
        rec("v7b", judge(trig * 0.01 if trig is not None else None, None))
        for name, va in variants.items():
            rec(name, judge(va.first_open(s["mix"], ref), None))
        if made % 20 == 0:
            print(f"[B {made}/{n_neg}] elapsed={time.time()-t_start:.0f}s", flush=True)

    print(f"\n== barge-in 对照（A {n_pos} 条 / B_only {n_neg} 条；{th_desc}）==")
    print(f"{'模型':14s} {'A打断召回':>9s} {'漏':>5s} {'B期间误触':>9s} "
          f"{'延迟p50':>8s} {'延迟p90':>8s} {'B_only误触':>10s}")
    summary = {}
    for name in ("v7b", "v4-3s1s", "v4-1.5s0.5s"):
        st = stats[name]
        n_a = st["ok"] + st["miss"] + st["false_pre"]
        recall = st["ok"] / max(1, n_a)
        d = sorted(st["delays"])
        p50 = d[len(d) // 2] if d else float("nan")
        p90 = d[int(len(d) * 0.9)] if d else float("nan")
        false_b = st["false"] / max(1, st["false"] + st["ok_closed"])
        summary[name] = {"a_recall": round(recall, 4), "a_miss": st["miss"],
                         "a_false_pre": st["false_pre"],
                         "delay_p50": round(p50, 2) if d else None,
                         "delay_p90": round(p90, 2) if d else None,
                         "b_only_false": round(false_b, 4)}
        print(f"{name:14s} {recall*100:8.1f}% {st['miss']:5d} "
              f"{st['false_pre']/max(1,n_a)*100:8.1f}% {p50:7.2f}s {p90:7.2f}s "
              f"{false_b*100:9.1f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": {"n_pos": n_pos, "n_neg": n_neg, "seed": args.seed,
                              "v4": th_desc, "scenario":
                              "B 从 0s 起, A 在 t0∈[1,3]s 开口, A 大 B 小 5~15dB"},
                   "summary": summary}, f, ensure_ascii=False, indent=1)
    print(f"\n明细: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
