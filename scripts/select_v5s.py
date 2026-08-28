# -*- coding: utf-8 -*-
"""v5s 微调选点: 对每个候选 checkpoint 报 val 两条件 e2e (src=v1 干净 / src=v2 增广)。

用 EMA-CMVN 特征 + torch 整段前向 (与流式数学等价), confirm=2 门控 + ±20 帧判定。
用法: python scripts/select_v5s.py ckpt1.pt [ckpt2.pt ...] [--max-n 500]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvad_common import ROOT, load_labels  # noqa: E402
from train_pvad import MODELS  # noqa: E402
from eval_stream import gate_trigger, judge  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--val-dir", default=str(ROOT / "data" / "mixtures_v4" / "val"))
    ap.add_argument("--max-n", type=int, default=500)
    ap.add_argument("--feats-subdir", default="feats_ema")
    args = ap.parse_args()

    td = Path(args.val_dir)
    recs = [r for r in load_labels(td / "labels.jsonl")
            if r.get("overlap_frames")][: args.max_n]
    print(f"val 双讲样本 {len(recs)} 条")

    for ckpt in args.ckpts:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = MODELS[state["cond"]]()
        model.load_state_dict(state["model"])
        model.eval()
        g = {"v1": {"ok": 0, "miss": 0, "false": 0},
             "v2": {"ok": 0, "miss": 0, "false": 0}}
        for r in recs:
            feats = np.load(td / args.feats_subdir / f"{r['id']}.npy")
            toks = np.load(td / "emb_tokens" / f"{r['id']}.npy")
            labels = np.asarray(r["labels"])
            T = min(len(feats), len(labels))
            x = torch.from_numpy(feats[:T]).unsqueeze(0)
            t = torch.from_numpy(toks).unsqueeze(0)
            km = torch.zeros(1, len(toks), dtype=torch.bool)
            with torch.no_grad():
                p2 = torch.softmax(model(x, t, km), -1)[0, :, 2].numpy()
            src = r.get("src", "v2")
            g[src][judge(gate_trigger(p2), r["overlap_frames"], labels[:T])] += 1
        name = Path(ckpt).name
        for src in ("v1", "v2"):
            s = g[src]
            n = max(s["ok"] + s["miss"] + s["false"], 1)
            print(f"{name} [{src}] 漏 {s['miss'] / n:.3f} 误 {s['false'] / n:.3f} "
                  f"正确 {s['ok'] / n:.3f} (n={n})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
