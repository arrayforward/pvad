# -*- coding: utf-8 -*-
"""同集比较 tse-gitee pvad_direct_v3/v4 与当前 pvad_v7b_stream。

两种句级场景（每条件各 --max-n 条；按项目验收口径不再测试错误注册 C）：
  A+B 注册 A：放行率；only B 注册 A：关闭率。
v7b 另报当前双讲 e2e 判定（重叠区 ±20 帧），避免把“任意位置触发”误当正确。

注意：direct 系列原训练/发布指标来自 direct_scene_v2 阵列参考通道。本脚本把其
单声道编码路径原样用于当前 mixtures，属于当前部署域对照，不能复现官方指标。
`--direct-version v3|v4` 选择 TSE 权重，默认 v4。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

DV = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DV / "scripts"))
from pvad_common import DATA_ROOT, data_path, fbank, load_labels, read_wav  # noqa: E402

TSE_CODE = Path(r"D:/ai-app/tse-gitee/code")
sys.path.insert(0, str(TSE_CODE))
sys.path.insert(0, str(TSE_CODE / "scripts"))
from tse_array.config import TSEConfig  # noqa: E402
from tse_array.speaker_multiscale import MultiscaleSpeakerEncoder  # noqa: E402
from train_multiscale_confidence_pvad import (  # noqa: E402
    ConfidenceProjection, confidence_score, score, window_bounds)


def p2_stream(sess, raw, emb, m0, chunk=5):
    """当前生产流式接口：EMA-CMVN alpha=0.02 + 外置 GRU state。"""
    m = m0.copy()
    feats = np.empty_like(raw)
    for i in range(len(raw)):
        m = 0.98 * m + 0.02 * raw[i]
        feats[i] = raw[i] - m
    h = np.zeros((2, 1, 128), dtype=np.float32)
    outs = []
    for s in range(0, len(feats), chunk):
        lg, h = sess.run(None, {
            "feats_chunk": feats[None, s:s + chunk].astype(np.float32),
            "emb": emb[None].astype(np.float32), "h0": h})
        outs.append(lg[0])
    lg = np.concatenate(outs, axis=0)
    e = np.exp(lg - lg.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True))[:, 2]


def calibrate_level(pcm, target):
    """与 C++ agc_normalize 同口径的整段恒定增益；target<=0 时关闭。"""
    if target <= 0:
        return pcm
    x = pcm.astype(np.float32)
    frames = x[:len(x) // 160 * 160].reshape(-1, 160).astype(np.float64)
    frms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-20)
    active = frames[frms > 0.003]
    power = float(np.mean(active ** 2)) if len(active) else float(np.mean(frames ** 2))
    gain = float(np.clip(target / max(np.sqrt(power), 1e-6), 0.1, 12.0))
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 0:
        gain = min(gain, 0.95 / peak)
    return np.clip(x * gain, -0.98, 0.98).astype(np.float32)


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
    if trig is None:
        return "miss"
    for st, ed in overlaps:
        if st - 20 <= trig <= ed + 20:
            return "ok"
    return "ok" if labels[min(trig, len(labels) - 1)] == 2 else "false"


class DirectGateModel:
    def __init__(self, version="v3"):
        torch.set_num_threads(4)
        self.name = f"direct_{version}"
        self.cfg = TSEConfig()
        self.enc = MultiscaleSpeakerEncoder(self.cfg).eval()
        self.enc.load_state_dict(torch.load(
            TSE_CODE / "models/speaker_multiscale_distill_e50/best.pt",
            map_location="cpu", weights_only=False)["model"])
        st = torch.load(TSE_CODE / f"models/pvad_direct_{version}/best.pt",
                        map_location="cpu", weights_only=False)
        self.gate = ConfidenceProjection().eval()
        self.gate.load_state_dict(st["model"])
        self.th = float(st["threshold"])
        self.unc = float(st.get("uncertain_threshold", 0.3))

    @torch.inference_mode()
    def embed(self, wave):
        return self.enc(torch.from_numpy(wave.astype(np.float32))[None])[0].numpy()

    @torch.inference_mode()
    def windows(self, wave):
        w, h = int(3.0 * self.cfg.sr), int(1.0 * self.cfg.sr)
        bounds = window_bounds(len(wave), w, h)
        segs = np.zeros((len(bounds), w), dtype=np.float32)
        for i, (s, e) in enumerate(bounds):
            segs[i, :e - s] = wave[s:e]
        return self.enc(torch.from_numpy(segs))

    @torch.inference_mode()
    def decide(self, windows, reference):
        ident = float(score(self.gate, windows, reference))
        conf = float(torch.sigmoid(confidence_score(self.gate, windows, reference)))
        closed = bool((ident < self.th) and (conf >= self.unc))
        return ident, conf, closed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=200)
    ap.add_argument("--only", choices=["clean", "aug", "both"], default="both")
    ap.add_argument("--direct-version", choices=["v3", "v4"], default="v4")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--direct-agc", type=float, default=0.0,
                    help="direct 输入有效 RMS 校准目标，0=关闭；只影响 direct，不改变 v7b")
    ap.add_argument("--out", default=str(DV / "build" / "pvad_compare_tse.json"))
    args = ap.parse_args()

    direct = DirectGateModel(args.direct_version)
    sess = ort.InferenceSession(str(DV / "models/pvad/pvad_v7b_stream.onnx"),
                                providers=["CPUExecutionProvider"])
    emb_cache, m0_cache = {}, {}

    def direct_emb(path):
        key = str(path)
        if key not in emb_cache:
            pcm, sr = read_wav(path)
            assert sr == 16000
            emb_cache[key] = direct.embed(pcm)
        return emb_cache[key]

    def prior(path):
        key = str(path)
        if key not in m0_cache:
            pcm, sr = read_wav(path)
            assert sr == 16000
            m0_cache[key] = fbank(pcm.astype(np.float32)).astype(np.float64).mean(axis=0)
        return m0_cache[key]

    report = {"config": {"max_n_per_scenario": args.max_n, "split": args.split,
                         "direct_version": direct.name, "direct_agc": args.direct_agc,
                         "direct_threshold": direct.th, "direct_uncertain": direct.unc,
                         "v7b_gate": "P>0.5 confirm=2 hyst=0.2, EMA prior, no VAD/warmup"},
              "conditions": {}}
    all_rows = {}
    for tag, rel in (("clean", f"mixtures/{args.split}"), ("aug", f"mixtures_v2/{args.split}")):
        if args.only != "both" and args.only != tag:
            continue
        td = DATA_ROOT / rel
        recs = load_labels(td / "labels.jsonl")
        pos = [r for r in recs if r.get("overlap_frames")][:args.max_n]
        neg = [r for r in recs if not r.get("overlap_frames")
               and 2 not in r["labels"]][:args.max_n]
        stats = {m: {"open_a": 0, "close_b": 0,
                     "ok": 0, "miss": 0, "false": 0}
                 for m in (direct.name, "v7b")}
        rows = []
        t0 = time.time()

        def run_one(r, scenario):
            pcm, sr = read_wav(data_path(r["path"]))
            assert sr == 16000
            enroll_path = data_path(r["enrollment"])
            direct_ref = direct_emb(enroll_path)
            cam_ref = np.load(td / "emb" / f"{r['id']}.npy").astype(np.float32)
            m0 = prior(enroll_path)
            windows = direct.windows(calibrate_level(pcm, args.direct_agc))
            ident, conf, closed = direct.decide(windows, direct_ref)
            direct_open = not closed
            raw = fbank(pcm.astype(np.float32)).astype(np.float64)
            trig = gate_trigger(p2_stream(sess, raw, cam_ref, m0))
            v7b_open = trig is not None
            if scenario == "A":
                stats[direct.name]["open_a"] += direct_open
                stats["v7b"]["open_a"] += v7b_open
                verdict = judge(trig, r["overlap_frames"], np.asarray(r["labels"]))
                stats["v7b"][verdict] += 1
            else:
                stats[direct.name]["close_b"] += not direct_open
                stats["v7b"]["close_b"] += not v7b_open
            return {"id": r["id"], "scenario": scenario, "direct_ident": round(ident, 5),
                    "direct_conf": round(conf, 5), "direct_open": direct_open,
                    "v7b_trigger": trig, "v7b_open": v7b_open}

        for i, r in enumerate(pos):
            rows.append(run_one(r, "A"))
            if (i + 1) % 25 == 0:
                print(f"[{tag}] A {i + 1}/{len(pos)} elapsed={time.time()-t0:.0f}s", flush=True)
        for i, r in enumerate(neg):
            rows.append(run_one(r, "B"))
            if (i + 1) % 50 == 0:
                print(f"[{tag}] B {i + 1}/{len(neg)} elapsed={time.time()-t0:.0f}s", flush=True)

        summary = {}
        for model, s in stats.items():
            summary[model] = {
                "open_a": s["open_a"] / len(pos), "close_b": s["close_b"] / len(neg),
                "e2e_ok": s["ok"] / len(pos), "e2e_miss": s["miss"] / len(pos),
                "e2e_false": s["false"] / len(pos)}
        report["conditions"][tag] = {"n_positive": len(pos), "n_only_b": len(neg),
                                     "metrics": summary}
        all_rows[tag] = rows
        print(f"\n== {tag}: A {len(pos)} 条 / B {len(neg)} 条 ==")
        for model, m in summary.items():
            print(f"{model:10s} 放行A {m['open_a']*100:5.1f}% "
                  f"关B {m['close_b']*100:5.1f}% | e2e {m['e2e_ok']*100:5.1f}% "
                  f"(漏 {m['e2e_miss']*100:.1f}% 误 {m['e2e_false']*100:.1f}%)")
    report["rows"] = all_rows
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\n明细与汇总: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
