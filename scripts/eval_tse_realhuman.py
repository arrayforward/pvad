# -*- coding: utf-8 -*-
"""TSE (erb_df_v2_e16) 真人验收评估：real_test_tse 100 条（全真人 test 池）。

每条经单麦流式链（ErbDfStreamer input_mics=1，内置前 2s 电平校准）：
  A_mix (70): SI-SDR(in)->SI-SDR(out)（对混合前干净 A 轨，延迟搜索对齐），
              B 泄漏 = 增强输出在纯 B 帧的能量 / 混合中 B 分量在该帧能量；
              下游门控（直接作用于 TSE 输出）放行率 —— 越高越好。
  B_only (30): 抑制比 = 20log10(RMS(enh)/RMS(mix))，越负越好；
              下游门控关闭率 —— 越高越好。
对照：同一批门控直接作用于未分离的 mix（配对基线）。

用法:
  python scripts/eval_tse_realhuman.py [--th4 0.40] --out build/eval_tse_realhuman.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

DV = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DV / "scripts"))
from pvad_common import DATA_ROOT, fbank, read_wav  # noqa: E402
from eval_realhuman import Gates  # noqa: E402

TSE_CODE = Path(r"D:/ai-app/tse-gitee/code")
sys.path.insert(0, str(TSE_CODE))
from tse_array.config import TSEConfig  # noqa: E402
from tse_array.erb_df_model import ErbDfTSE  # noqa: E402
from tse_array.speaker_multiscale import (MultiscaleSpeakerEncoder,  # noqa: E402
                                          enroll_tse_embedding)
from tse_array.streaming import ErbDfStreamer  # noqa: E402

SR = 16000


def si_sdr_align(est, ref, max_delay=2048):
    """尺度不变 SDR，先在 [0, max_delay] 内搜索 est 相对 ref 的整数延迟。"""
    est = np.asarray(est, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    best, best_d = -1e30, 0
    # 粗搜 16 采样步长再细搜 ±16：est 延迟 d 对应 est[d:] 对齐 ref[:n-d]
    for d in range(0, min(max_delay, n - SR // 2), 16):
        c = float(np.dot(est[d:], ref[:n - d]))
        if c > best:
            best, best_d = c, d
    d0 = best_d
    for d in range(max(0, d0 - 16), min(max_delay, n - SR // 2, d0 + 17)):
        c = float(np.dot(est[d:], ref[:n - d]))
        if c > best:
            best, best_d = c, d
    d = best_d
    e, r = est[d:], ref[:n - d]
    alpha = float(np.dot(e, r) / (np.dot(r, r) + 1e-12))
    tgt = alpha * r
    err = e - tgt
    return 10 * np.log10(np.dot(tgt, tgt) / (np.dot(err, err) + 1e-12) + 1e-12), d


class MonoTSE:
    def __init__(self, ckpt):
        self.cfg = TSEConfig()
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = ErbDfTSE(self.cfg, multi_level=True, mid_dim=4096).eval()
        self.model.load_state_dict(st["model"])

    @torch.no_grad()
    def extract(self, mono, emb):
        st = ErbDfStreamer(self.model, self.cfg, emb,
                           chunk_frames=self.cfg.chunk_frames, input_mics=1)
        outs = []
        step = self.cfg.chunk_frames * self.cfg.hop
        for off in range(0, len(mono), step):
            outs.append(st.process_chunk(mono[None, off:off + step]))
        outs.append(st.flush())
        return np.concatenate(outs)


def frame_mask(labels, value, n):
    m = np.zeros(n, dtype=bool)
    for i, lb in enumerate(labels):
        if lb == value:
            m[i * 160:min((i + 1) * 160, n)] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=str(DATA_ROOT / "real_test_tse"))
    ap.add_argument("--th4", type=float, default=None)
    ap.add_argument("--unc4", type=float, default=None)
    ap.add_argument("--th3", type=float, default=None)
    ap.add_argument("--unc3", type=float, default=None)
    ap.add_argument("--max-n", type=int, default=0, help="只评前 N 条（冒烟用）")
    ap.add_argument("--out", default=str(DV / "build" / "eval_tse_realhuman.json"))
    args = ap.parse_args()

    tdir = Path(args.set)
    recs = [json.loads(l) for l in open(tdir / "meta.jsonl", encoding="utf-8") if l.strip()]
    if args.max_n:
        pos = [r for r in recs if r["scenario"] == "A_mix"][:args.max_n]
        neg = [r for r in recs if r["scenario"] == "B_only"][:max(1, args.max_n // 2)]
        recs = pos + neg
    print(f"测试集 {tdir} 共 {len(recs)} 条")

    tse = MonoTSE(TSE_CODE / "models/erb_df_v2_e16/best.pt")
    enc = MultiscaleSpeakerEncoder(tse.cfg).eval()
    enc.load_state_dict(torch.load(
        TSE_CODE / "models/speaker_multiscale_distill_e50/best.pt",
        map_location="cpu", weights_only=False)["model"])
    gates = Gates(th4=args.th4, unc4=args.unc4, th3=args.th3, unc3=args.unc3)
    print(f"门控: v3 TH={gates.v3.th} | v4 TH={gates.v4.th} UNC={gates.v4.unc} | v7b")

    rows = []
    t0 = time.time()
    for i, r in enumerate(recs):
        mix, sr = read_wav(tdir / f"{r['id']}_mix.wav")
        tgt, _ = read_wav(tdir / f"{r['id']}_target.wav")
        enroll, _ = read_wav(tdir / f"{r['id']}_enroll.wav")
        assert sr == SR
        with torch.inference_mode():
            emb = enroll_tse_embedding(enc, enroll.astype(np.float32), tse.cfg)
            ref192 = enc(torch.from_numpy(enroll.astype(np.float32))[None])[0].numpy()
        enh = tse.extract(mix.astype(np.float32), emb)

        row = {"id": r["id"], "scenario": r["scenario"], "snr_db": r["snr_db"]}
        if r["scenario"] == "A_mix":
            si_in, _ = si_sdr_align(mix, tgt)
            si_out, delay = si_sdr_align(enh, tgt)
            # B 泄漏：纯 B 帧（labels==1）中 enh 能量 / 混合中 B 分量能量
            n = min(len(enh) - delay, len(mix))
            b_comp = (mix - tgt)[:n]
            enh_al = enh[delay:delay + n]
            bm = frame_mask(r["labels"], 1, n)
            leak = np.nan
            if bm.any() and np.sqrt(np.mean(b_comp[bm] ** 2)) > 1e-6:
                leak = 20 * np.log10(
                    np.sqrt(np.mean(enh_al[bm] ** 2)) /
                    np.sqrt(np.mean(b_comp[bm] ** 2)) + 1e-12)
            row.update(si_in=round(si_in, 2), si_out=round(si_out, 2),
                       si_imp=round(si_out - si_in, 2),
                       b_leak_db=None if np.isnan(leak) else round(float(leak), 1))
        else:
            sup = 20 * np.log10(np.sqrt(np.mean(enh ** 2)) /
                                (np.sqrt(np.mean(mix ** 2)) + 1e-12) + 1e-12)
            row["suppress_db"] = round(float(sup), 1)

        # 下游门控：TSE 输出 vs 原始 mix（配对）
        cam_emb = gates.cam.embed(enroll.astype(np.float32))
        m0 = fbank(enroll.astype(np.float32)).astype(np.float64).mean(axis=0)
        for tag, wave in (("tse", enh), ("raw", mix)):
            win = gates.windows(wave.astype(np.float32))
            _, _, o3 = gates.direct(gates.v3, win, ref192)
            _, _, o4 = gates.direct(gates.v4, win, ref192)
            p2, trig, o7 = gates.v7b(wave.astype(np.float32), cam_emb, m0)
            row[f"v3_{tag}"], row[f"v4_{tag}"], row[f"v7b_{tag}"] = o3, o4, o7
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"[{i + 1}/{len(recs)}] elapsed={time.time()-t0:.0f}s", flush=True)

    def agg(rows, key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return {"mean": round(float(np.mean(v)), 2),
                "p50": round(float(np.median(v)), 2),
                "min": round(float(np.min(v)), 2),
                "max": round(float(np.max(v)), 2)}

    am = [r for r in rows if r["scenario"] == "A_mix"]
    bo = [r for r in rows if r["scenario"] == "B_only"]
    summary = {"A_mix": {"n": len(am), "si_in": agg(am, "si_in"),
                         "si_out": agg(am, "si_out"), "si_imp": agg(am, "si_imp"),
                         "b_leak_db": agg(am, "b_leak_db")},
               "B_only": {"n": len(bo), "suppress_db": agg(bo, "suppress_db")}}
    for tag in ("raw", "tse"):
        for m in ("v3", "v4", "v7b"):
            summary["A_mix"][f"open_{m}_{tag}"] = round(
                np.mean([r[f"{m}_{tag}"] for r in am]), 4)
            summary["B_only"][f"open_{m}_{tag}"] = round(
                np.mean([r[f"{m}_{tag}"] for r in bo]), 4)

    print("\n== A_mix (70): 提取质量 ==")
    print(f"SI-SDR in {summary['A_mix']['si_in']['mean']:+.2f} dB -> "
          f"out {summary['A_mix']['si_out']['mean']:+.2f} dB "
          f"(提升 {summary['A_mix']['si_imp']['mean']:+.2f}, "
          f"最差 {summary['A_mix']['si_imp']['min']:+.2f})")
    print(f"B 泄漏(纯B帧): 均值 {summary['A_mix']['b_leak_db']['mean']:+.1f} dB, "
          f"最差 {summary['A_mix']['b_leak_db']['max']:+.1f} dB")
    print("== B_only (30): 抑制 ==")
    print(f"抑制比 均值 {summary['B_only']['suppress_db']['mean']:+.1f} dB, "
          f"最差 {summary['B_only']['suppress_db']['max']:+.1f} dB")
    print("== 下游门控放行率（A_mix 要高 / B_only 要低）==")
    for tag, name in (("raw", "原始mix"), ("tse", "TSE输出")):
        print(f"  {name:8s}: " + " | ".join(
            f"{m} A {summary['A_mix'][f'open_{m}_{tag}']*100:5.1f}% "
            f"B {summary['B_only'][f'open_{m}_{tag}']*100:5.1f}%"
            for m in ("v3", "v4", "v7b")))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": {"set": str(tdir), "v4_th": gates.v4.th,
                              "v3_th": gates.v3.th, "v4_unc": gates.v4.unc},
                   "summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
    print(f"\n明细: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
