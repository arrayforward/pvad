# -*- coding: utf-8 -*-
"""真人数据端到端评估：A 大 B 小部署场景下的 PVAD 门控对照。

动机：此前“标准 B”单用例用了 SAPI TTS（voice2/tts_ref/story_tts），不代表真人
干扰；mixtures/test 虽然全是真人（aishell/stcmds/primewords），但 SNR -5~10dB
包含 B 比 A 还响的样本，不符合“A 近场大、B 远场小”的实际部署。

本脚本全部使用 split.json test 池真人语音（训练未见），内存合成不落盘：
  A_mix  : A（有效 RMS 归一到 ~0.03-0.06）+ B 低 5~15dB，注册 A → 期望放行
  B_only : 只有 B（与混合中同等的低电平），注册 A → 期望关闭
  A_only : 只有 A（近场电平），注册 A → 期望放行（ sanity ）

模型：direct_v3 / direct_v4（TSE 仓 ConfidenceProjection 门控，可 --th4 改 v4 阈值）
和当前 pvad_v7b_stream（P>0.5 confirm=2）。

用法:
  python scripts/eval_realhuman.py --n 200 --out build/eval_realhuman.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

DV = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DV / "scripts"))
from pvad_common import DATA_ROOT, CampplusEmbedder, data_path, fbank, read_wav  # noqa: E402
from gen_mixtures import build_track, frame_labels, load_manifest, rms  # noqa: E402
from compare_tse_pvad import DirectGateModel, gate_trigger, p2_stream  # noqa: E402

SR = 16000


def active_rms(x, mask):
    seg = x[mask == 1]
    return rms(seg) if len(seg) else rms(x)


class Gates:
    """共享 speaker encoder 的 direct_v3/v4 门控 + v7b。"""

    def __init__(self, th4=None, unc4=None, th3=None, unc3=None):
        torch.set_num_threads(4)
        self.v3 = DirectGateModel("v3")
        self.v4 = DirectGateModel("v4")
        if th4 is not None:
            self.v4.th = float(th4)
        if unc4 is not None:
            self.v4.unc = float(unc4)
        if th3 is not None:
            self.v3.th = float(th3)
        if unc3 is not None:
            self.v3.unc = float(unc3)
        # 两个 gate 共用同一 speaker encoder，窗嵌入只算一次
        self.enc = self.v4.enc
        self.cfg = self.v4.cfg
        self.sess = ort.InferenceSession(str(DV / "models/pvad/pvad_v7b_stream.onnx"),
                                         providers=["CPUExecutionProvider"])
        self.cam = CampplusEmbedder()

    @torch.inference_mode()
    def windows(self, wave):
        w, h = int(3.0 * self.cfg.sr), int(1.0 * self.cfg.sr)
        n = max(1, 1 + max(0, len(wave) - w) // h)
        segs = np.zeros((n, w), dtype=np.float32)
        for i in range(n):
            s = i * h
            segs[i, :max(0, min(w, len(wave) - s))] = wave[s:s + w]
        return self.enc(torch.from_numpy(segs))

    @torch.inference_mode()
    def direct(self, gate, windows, ref):
        from train_multiscale_confidence_pvad import confidence_score, score
        ident = float(score(gate.gate, windows, ref))
        conf = float(torch.sigmoid(confidence_score(gate.gate, windows, ref)))
        closed = bool((ident < gate.th) and (conf >= gate.unc))
        return ident, conf, not closed

    def v7b(self, pcm, emb, m0):
        raw = fbank(pcm.astype(np.float32)).astype(np.float64)
        p2 = p2_stream(self.sess, raw, emb, m0)
        trig = gate_trigger(p2)
        return p2, trig, trig is not None


def gen_sample(rng, by_spk, speakers, target_rms):
    """返回 dict(wave, labels, enroll_path, cam 用不到此处) 或 None。"""
    tgt_spk, itf_spk = rng.sample(speakers, 2)
    length = int(rng.uniform(3.0, 8.0) * SR)
    utts_t = by_spk[tgt_spk]
    # 注册句规则与 gen_mixtures/生产一致：目标说话人其它 3-10s 句子
    cands = [u for u in utts_t if 3.0 <= u[1] <= 10.0]
    if not cands:
        return None
    enroll = rng.choice(cands)[0]
    pool_t = [u for u in utts_t if u[0] != enroll]
    if len(pool_t) < 1:
        return None
    tgt, mask_t, _ = build_track(rng, pool_t, length)
    itf, mask_g, _ = build_track(rng, by_spk[itf_spk], length)
    if not (mask_t == 1).any() or not (mask_g == 1).any():
        return None
    snr_db = rng.uniform(5.0, 15.0)  # A 比 B 大 5~15dB（近场用户 vs 远场干扰）
    a_r = active_rms(tgt, mask_t)
    b_r = active_rms(itf, mask_g)
    ga = target_rms / a_r
    gb = ga * a_r / b_r / 10 ** (snr_db / 20.0)
    mix = np.clip(tgt * ga + itf * gb, -0.98, 0.98).astype(np.float32)
    return {"tgt_spk": tgt_spk, "itf_spk": itf_spk, "enroll": enroll,
            "mix": mix, "a_only": np.clip(tgt * ga, -0.98, 0.98).astype(np.float32),
            "b_only": np.clip(itf * gb, -0.98, 0.98).astype(np.float32),
            "labels": frame_labels(mask_t, mask_g),
            "snr_db": round(snr_db, 2), "level": target_rms,
            "a_rms": round(a_r * ga, 5), "b_rms": round(b_r * gb, 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="每场景样本数")
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--th4", type=float, default=None,
                    help="覆盖 v4 阈值（默认用 checkpoint 自带，发布值 0.23）")
    ap.add_argument("--unc4", type=float, default=None)
    ap.add_argument("--th3", type=float, default=None)
    ap.add_argument("--unc3", type=float, default=None)
    ap.add_argument("--out", default=str(DV / "build" / "eval_realhuman.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_spk = load_manifest(DATA_ROOT / "manifest.jsonl")
    with open(DATA_ROOT / "split.json", encoding="utf-8") as f:
        pools = json.load(f)
    allow = set(pools["test"])
    speakers = [s for s in pools["test"] if s in by_spk and len(by_spk[s]) >= 3]
    print(f"test 池真人说话人 {len(speakers)} / {len(allow)}")

    gates = Gates(th4=args.th4, unc4=args.unc4, th3=args.th3, unc3=args.unc3)
    print(f"direct_v3 TH={gates.v3.th} UNC={gates.v3.unc} | "
          f"direct_v4 TH={gates.v4.th} UNC={gates.v4.unc}")

    models = ["direct_v3", "direct_v4", "v7b"]
    stats = {sc: {m: {"open": 0, "n": 0} for m in models}
             for sc in ("A_mix", "B_only", "A_only")}
    rows = []
    t0 = time.time()
    made = 0
    while made < args.n:
        level = float(np.clip(rng.gauss(0.045, 0.008), 0.03, 0.06))
        s = gen_sample(rng, by_spk, speakers, level)
        if s is None:
            continue
        made += 1
        enroll_pcm, sr = read_wav(s["enroll"])
        assert sr == SR
        with torch.inference_mode():
            ref = gates.enc(torch.from_numpy(
                enroll_pcm.astype(np.float32))[None])[0].numpy()
        cam_emb = gates.cam.embed(enroll_pcm.astype(np.float32))
        m0 = fbank(enroll_pcm.astype(np.float32)).astype(np.float64).mean(axis=0)

        for sc, wave in (("A_mix", s["mix"]), ("B_only", s["b_only"]),
                         ("A_only", s["a_only"])):
            win = gates.windows(wave)
            i3, c3, o3 = gates.direct(gates.v3, win, ref)
            i4, c4, o4 = gates.direct(gates.v4, win, ref)
            p2, trig, o7 = gates.v7b(wave, cam_emb, m0)
            for m, o in (("direct_v3", o3), ("direct_v4", o4), ("v7b", o7)):
                stats[sc][m]["open"] += int(o)
                stats[sc][m]["n"] += 1
            rows.append({"id": made, "scenario": sc, "tgt": s["tgt_spk"],
                         "itf": s["itf_spk"], "snr_db": s["snr_db"],
                         "a_rms": s["a_rms"], "b_rms": s["b_rms"],
                         "v3_ident": round(i3, 4), "v3_conf": round(c3, 4),
                         "v3_open": o3,
                         "v4_ident": round(i4, 4), "v4_conf": round(c4, 4),
                         "v4_open": o4, "v7b_trigger": trig, "v7b_open": o7,
                         "v7b_p2": [round(float(x), 3) for x in p2]})
        if made % 20 == 0:
            print(f"[{made}/{args.n}] elapsed={time.time()-t0:.0f}s", flush=True)

    summary = {}
    print(f"\n== 真人 test 池, 每场景 {args.n} 条, A 大 B 小 (SNR 5~15dB) ==")
    am = [r for r in rows if r["scenario"] == "A_mix"]
    print(f"电平审计: A 有效RMS {min(r['a_rms'] for r in am):.4f}~"
          f"{max(r['a_rms'] for r in am):.4f}, B 有效RMS "
          f"{min(r['b_rms'] for r in am):.4f}~{max(r['b_rms'] for r in am):.4f}, "
          f"SNR {min(r['snr_db'] for r in am):.1f}~{max(r['snr_db'] for r in am):.1f}dB")
    for sc in ("A_mix", "B_only", "A_only"):
        summary[sc] = {}
        for m in models:
            st = stats[sc][m]
            rate = st["open"] / max(1, st["n"])
            summary[sc][m] = {"open_rate": round(rate, 4), "n": st["n"]}
        want = "放行" if sc.startswith("A") else "关闭"
        line = f"{sc:7s} 期望{want}: "
        for m in models:
            st = stats[sc][m]
            rate = st["open"] / max(1, st["n"])
            v = rate if sc.startswith("A") else 1 - rate
            line += f"{m} {v*100:5.1f}% | "
        print(line)

    report = {"config": {"n_per_scenario": args.n, "seed": args.seed,
                         "scenario": "真人 test 池, A 有效RMS 0.03-0.06, B 低 5~15dB",
                         "v3_th": gates.v3.th, "v4_th": gates.v4.th,
                         "v4_unc": gates.v4.unc},
              "summary": summary, "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\n明细: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
