# -*- coding: utf-8 -*-
"""PVAD 评估: 帧级指标 (按 SNR 分档) + 端到端双讲门控对照 (vs CAM++ AS-norm)。

用法:
  python scripts/eval_pvad.py --frames            # 帧级指标 (test 全集)
  python scripts/eval_pvad.py --e2e [--max-n 500] # 端到端双讲对照
  python scripts/eval_pvad.py --all
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvad_common import ROOT, load_labels  # noqa: E402
from train_pvad import MODELS, MixtureDataset, collate, FEAT_DIM, EMB_DIM  # noqa: E402


def load_model(ckpt=None):
    ckpt = ckpt or ROOT / "models" / "pvad" / "best.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cond = state.get("cond", "concat")
    model = MODELS[cond]()
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def snr_bucket(snr):
    if snr < 0:
        return "<0dB"
    if snr <= 5:
        return "0-5dB"
    return ">5dB"


@torch.no_grad()
def frame_metrics(model, test_dir, feats_subdir="feats"):
    use_tokens = getattr(model, "USE_TOKENS", False)
    ds = MixtureDataset(test_dir, use_tokens=use_tokens, feats_subdir=feats_subdir)
    ld = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate,
                    num_workers=0)
    # 按 SNR 分档统计: bucket -> [tp, fp, fn, far_n, far_d, correct, total]
    buckets = {}
    i = 0
    for x, e, kv_mask, y in ld:
        logits = model(x, e, kv_mask) if use_tokens else model(x, e)
        pred = logits.argmax(-1)
        B = x.shape[0]
        for b in range(B):
            snr = ds.recs[i]["snr_db"]
            i += 1
            bk = buckets.setdefault(snr_bucket(snr), np.zeros(7, dtype=np.int64))
            mask = y[b] != -100
            t, p = y[b][mask], pred[b][mask]
            bk[0] += ((t == 2) & (p == 2)).sum()
            bk[1] += ((t != 2) & (p == 2)).sum()
            bk[2] += ((t == 2) & (p != 2)).sum()
            bk[3] += ((t == 1) & (p == 2)).sum()
            bk[4] += (t == 1).sum()
            bk[5] += (t == p).sum()
            bk[6] += mask.sum()
    print(f"\n=== 帧级指标 (test, {len(ds)} 条) ===")
    print(f"{'SNR':<8}{'recall':>8}{'prec':>8}{'F1':>8}{'FAR':>8}{'acc':>8}")
    for name in ["<0dB", "0-5dB", ">5dB", "ALL"]:
        if name == "ALL":
            bk = sum(buckets.values())
        elif name in buckets:
            bk = buckets[name]
        else:
            continue
        tp, fp, fn, far_n, far_d, correct, total = bk
        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * rec * prec / max(rec + prec, 1e-9)
        print(f"{name:<8}{rec:>8.4f}{prec:>8.4f}{f1:>8.4f}"
              f"{far_n / max(far_d, 1):>8.4f}{correct / max(total, 1):>8.4f}")


# ---------------- 端到端双讲对照 ----------------

def pvad_gate_trigger(p2, thr=0.5, hyst=0.2, confirm=2):
    """与 Gate 相同的迟滞 + confirm 逻辑, 作用于帧级目标概率序列。
    返回触发帧 index, 未触发返回 None。"""
    consec = 0
    for t, p in enumerate(p2):
        if p > thr:
            consec += 1
            if consec >= confirm:
                return t
        elif p < thr - hyst:
            consec = 0
    return None


class AsnormGate:
    """复现 gate.h: AS-norm z + margin, confirm=2 迟滞。"""

    def __init__(self, z_thr=3.0, margin=0.15, z_hyst=0.5, confirm=2):
        self.z_thr, self.margin = z_thr, margin
        self.z_hyst, self.confirm = z_hyst, confirm
        self.consec = 0
        self.triggered = False

    def update(self, s_a, s_neg, z):
        if z > self.z_thr and (s_a - s_neg) > self.margin:
            self.consec += 1
            if self.consec >= self.confirm and not self.triggered:
                self.triggered = True
                return True
        elif z < self.z_thr - self.z_hyst:
            self.consec = 0
            self.triggered = False
        return False


def tnorm(cohort, emb, topk=100):
    sims = np.sort(cohort @ emb)[::-1][:topk]
    mu = sims.mean()
    sigma = sims.std()
    return mu, max(sigma, 1e-6)


def e2e(args, model=None):
    """对 test 中含双讲区间的样本: PVAD 门控 vs CAM++ AS-norm 门控。"""
    from pvad_common import CampplusEmbedder, read_wav, fbank, mean_normalize
    if model is None:
        model, _ = load_model()
    embedder = CampplusEmbedder(intra_threads=4)

    recs = [r for r in load_labels(Path(args.test_dir) / "labels.jsonl")
            if r.get("overlap_frames")]
    if args.max_n:
        recs = recs[: args.max_n]
    print(f"双讲样本 {len(recs)} 条")

    # cohort: 其它 test enrollment embedding (impostor 池)
    cohort = []
    for r in load_labels(Path(args.test_dir) / "labels.jsonl")[:300]:
        p = Path(args.test_dir) / "emb" / f"{r['id']}.npy"
        if p.exists():
            cohort.append(np.load(p))
    cohort = np.stack(cohort)
    print(f"cohort {len(cohort)} 条")

    # 干扰说话人语句索引 (负模板用)
    import collections
    spk_utts = collections.defaultdict(list)
    with open(ROOT / "data" / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            spk_utts[r["speaker"]].append(ROOT / r["path"])

    WIN = 50  # 500ms = 50 帧
    stats = {"pvad": {"miss": 0, "false": 0, "ok": 0, "delay": []},
             "asnorm": {"miss": 0, "false": 0, "ok": 0, "delay": []}}
    for idx, r in enumerate(recs):
        feats = np.load(Path(args.test_dir) / args.feats_subdir / f"{r['id']}.npy")
        emb = np.load(Path(args.test_dir) / "emb" / f"{r['id']}.npy")
        labels = np.asarray(r["labels"])
        T = min(len(feats), len(labels))
        labels = labels[:T]
        overlaps = r["overlap_frames"]

        # ---- PVAD 门控 ----
        feats_t = torch.from_numpy(feats[:T]).unsqueeze(0)
        if getattr(model, "USE_TOKENS", False):
            toks = np.load(Path(args.test_dir) / "emb_tokens" / f"{r['id']}.npy")
            emb_t = torch.from_numpy(toks).unsqueeze(0)
            kv_mask = torch.zeros(1, len(toks), dtype=torch.bool)
            with torch.no_grad():
                p2 = torch.softmax(model(feats_t, emb_t, kv_mask), -1)[0, :, 2].numpy()
        else:
            emb_t = torch.from_numpy(emb).unsqueeze(0)
            with torch.no_grad():
                p2 = torch.softmax(model(feats_t, emb_t), -1)[0, :, 2].numpy()
        if args.median and args.median > 1:
            from scipy.signal import medfilt
            p2 = medfilt(p2, args.median)
        trig = pvad_gate_trigger(p2)
        judge(stats["pvad"], trig, overlaps, labels)

        # ---- CAM++ AS-norm 门控 (500ms 窗, 无重叠) ----
        pcm, _ = read_wav(ROOT / r["path"])
        enroll_emb = emb
        # 负模板: 干扰说话人的一条语句 embedding
        itf_utts = spk_utts[r["interferer_speaker"]]
        itf_pcm, _ = read_wav(itf_utts[0])
        try:
            neg_emb = embedder.embed(itf_pcm)
        except ValueError:
            neg_emb = enroll_emb
        gate = AsnormGate()
        trig_w = None
        n_win = (len(pcm) - 8000) // 8000 + 1 if len(pcm) >= 8000 else 0
        for w in range(n_win):
            seg = pcm[w * 8000:(w + 1) * 8000]
            try:
                we = embedder.embed(seg)
            except ValueError:
                continue
            s_a = float(np.dot(enroll_emb, we))
            s_neg = float(np.dot(neg_emb, we))
            mu, sigma = tnorm(cohort, we)
            z = (s_a - mu) / sigma
            if gate.update(s_a, s_neg, z):
                trig_w = w * WIN + WIN - 1  # 窗尾帧
                break
        judge(stats["asnorm"], trig_w, overlaps, labels)
        if (idx + 1) % 100 == 0:
            print(f"  进度 {idx + 1}/{len(recs)}")

    print(f"\n=== 端到端双讲对照 ({len(recs)} 条含重叠样本) ===")
    print(f"{'方法':<10}{'漏打断':>8}{'误打断':>8}{'正确':>8}{'平均延迟(帧)':>12}")
    for name, s in stats.items():
        n = s["miss"] + s["false"] + s["ok"]
        dl = np.mean(s["delay"]) if s["delay"] else float("nan")
        print(f"{name:<10}{s['miss'] / n:>8.3f}{s['false'] / n:>8.3f}"
              f"{s['ok'] / n:>8.3f}{dl:>12.1f}")


def judge(stat, trig, overlaps, labels):
    """trig: 触发帧或 None。触发帧落在重叠区(含前后 20 帧容忍)= 正确打断;
    触发处无目标语音 = 误打断; 未触发 = 漏打断。"""
    TOL = 20
    if trig is None:
        stat["miss"] += 1
        return
    for st, ed in overlaps:
        if st - TOL <= trig <= ed + TOL:
            stat["ok"] += 1
            stat["delay"].append(trig - st)
            return
    if labels[min(trig, len(labels) - 1)] == 2:
        stat["ok"] += 1
        return
    stat["false"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", action="store_true")
    ap.add_argument("--e2e", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--test-dir", default=str(ROOT / "data" / "mixtures" / "test"))
    ap.add_argument("--max-n", type=int, default=500)
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint 路径, 默认 models/pvad/best.pt")
    ap.add_argument("--median", type=int, default=0,
                    help="对 PVAD 帧级目标概率做 N 帧中值滤波后再入门控 (e2e)")
    ap.add_argument("--feats-subdir", default="feats")
    args = ap.parse_args()
    ckpt = Path(args.ckpt) if args.ckpt else None
    model, state = load_model(ckpt)
    print(f"checkpoint {ckpt or '默认'} epoch={state.get('epoch')}, "
          f"params={state.get('n_params'):,}")
    if args.frames or args.all or not args.e2e:
        frame_metrics(model, args.test_dir, feats_subdir=args.feats_subdir)
    if args.e2e or args.all:
        e2e(args, model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
