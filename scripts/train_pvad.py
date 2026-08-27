# -*- coding: utf-8 -*-
"""PVAD 训练: GRU(2x128) + Linear, 输入 [fbank(80) | enrollment_emb(192)] = 272 维。

用法: python scripts/train_pvad.py [--epochs 20] [--batch 32] [--max-train N]
输出: models/pvad/best.pt, models/pvad/train_log.json
损失: 加权 CE, 目标类(2)权重 4 —— 近似"2->非2 过抑制更重"的非对称损失。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvad_common import ROOT, load_labels  # noqa: E402

FEAT_DIM = 80
EMB_DIM = 192
NUM_CLASSES = 3
DEVICE = "cpu"


class PvadModel(nn.Module):
    """v1/v2 架构: emb broadcast 拼接 (cond="concat")。"""

    def __init__(self, in_dim=FEAT_DIM + EMB_DIM, hidden=128, layers=2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden, NUM_CLASSES)

    def forward(self, feats, emb):
        x = torch.cat(
            [feats, emb.unsqueeze(1).expand(-1, feats.shape[1], -1)], dim=-1)
        out, _ = self.gru(x)
        return self.fc(out)  # [B, T, 3]


class PvadModelFiLM(nn.Module):
    """FiLM 条件: emb -> 每层 GRU 输入的 (gamma, beta), 逐维缩放偏移 (cond="film")。"""

    def __init__(self, feat_dim=FEAT_DIM, emb_dim=EMB_DIM, hidden=128):
        super().__init__()
        self.film_in = nn.Linear(emb_dim, 2 * feat_dim)
        self.gru1 = nn.GRU(feat_dim, hidden, batch_first=True)
        self.film_h = nn.Linear(emb_dim, 2 * hidden)
        self.gru2 = nn.GRU(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, NUM_CLASSES)

    def forward(self, feats, emb):
        gb1 = self.film_in(emb).unsqueeze(1)
        g1, b1 = gb1[..., :feats.shape[-1]], gb1[..., feats.shape[-1]:]
        h = feats * (1.0 + g1) + b1
        h, _ = self.gru1(h)
        gb2 = self.film_h(emb).unsqueeze(1)
        g2, b2 = gb2[..., :h.shape[-1]], gb2[..., h.shape[-1]:]
        h = h * (1.0 + g2) + b2
        h, _ = self.gru2(h)
        return self.fc(h)


class PvadModelAttn(nn.Module):
    """交叉注意力条件: 帧特征作 query, emb 作单 token KV, 残差融合 (cond="attn")。"""

    def __init__(self, feat_dim=FEAT_DIM, emb_dim=EMB_DIM, hidden=128, heads=4):
        super().__init__()
        self.in_proj = nn.Linear(feat_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, kdim=emb_dim,
                                          vdim=emb_dim, batch_first=True)
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, NUM_CLASSES)

    def forward(self, feats, emb):
        kv = emb.unsqueeze(1)  # [B, 1, 192]
        a, _ = self.attn(self.in_proj(feats), kv, kv)
        h = self.in_proj(feats) + a
        out, _ = self.gru(h)
        return self.fc(out)


MODELS = {"concat": PvadModel, "film": PvadModelFiLM, "attn": PvadModelAttn}


class MixtureDataset(Dataset):
    def __init__(self, mix_dir, max_n=None, feats_subdir="feats"):
        d = Path(mix_dir)
        self.dir = d
        self.feat_dir = d / feats_subdir
        self.recs = load_labels(d / "labels.jsonl")
        if max_n:
            import random
            random.Random(1234).shuffle(self.recs)  # 固定种子, 保证正负样本都入样
            self.recs = self.recs[:max_n]
        # 过滤缺失特征
        self.recs = [r for r in self.recs
                     if (self.feat_dir / f"{r['id']}.npy").exists()
                     and (d / "emb" / f"{r['id']}.npy").exists()]

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        feats = np.load(self.feat_dir / f"{r['id']}.npy")
        emb = np.load(self.dir / "emb" / f"{r['id']}.npy")
        labels = np.asarray(r["labels"], dtype=np.int64)
        T = min(len(feats), len(labels))
        return feats[:T], emb, labels[:T], r["snr_db"]


def collate(batch):
    feats, embs, labels, snrs = zip(*batch)
    B = len(batch)
    T = max(f.shape[0] for f in feats)
    x = torch.zeros(B, T, FEAT_DIM)
    e = torch.zeros(B, EMB_DIM)
    y = torch.full((B, T), -100, dtype=torch.long)
    for i, (f, em, l) in enumerate(zip(feats, embs, labels)):
        t = f.shape[0]
        x[i, :t] = torch.from_numpy(f)
        e[i] = torch.from_numpy(em)
        y[i, :t] = torch.from_numpy(l)
    return x, e, y


@torch.no_grad()
def evaluate(model, loader, class_weight):
    model.eval()
    tot_loss, tot_n = 0.0, 0
    # 帧级统计
    tp = fp = fn = 0        # 目标类(2)
    far_n = far_d = 0       # true 1 -> pred 2
    correct = 0
    crit = nn.CrossEntropyLoss(weight=class_weight, ignore_index=-100, reduction="sum")
    for x, e, y in loader:
        logits = model(x, e)
        tot_loss += crit(logits.reshape(-1, NUM_CLASSES), y.reshape(-1)).item()
        mask = y != -100
        pred = logits.argmax(-1)
        t, p = y[mask], pred[mask]
        tot_n += mask.sum().item()
        correct += (t == p).sum().item()
        tp += ((t == 2) & (p == 2)).sum().item()
        fp += ((t != 2) & (p == 2)).sum().item()
        fn += ((t == 2) & (p != 2)).sum().item()
        far_d += (t == 1).sum().item()
        far_n += ((t == 1) & (p == 2)).sum().item()
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * rec * prec / max(rec + prec, 1e-9)
    return {
        "loss": tot_loss / max(tot_n, 1),
        "acc": correct / max(tot_n, 1),
        "target_recall": rec,
        "target_precision": prec,
        "target_f1": f1,
        "far": far_n / max(far_d, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default=str(ROOT / "data" / "mixtures" / "train"))
    ap.add_argument("--val-dir", default=str(ROOT / "data" / "mixtures" / "val"))
    ap.add_argument("--out-dir", default=str(ROOT / "models" / "pvad"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target-weight", type=float, default=4.0)
    ap.add_argument("--weight1", type=float, default=1.0,
                    help="非目标类(1)权重, 提高可压 FAR")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt-name", default="best.pt")
    ap.add_argument("--log-name", default="train_log.json")
    ap.add_argument("--cond", choices=["concat", "film", "attn"], default="concat",
                    help="enrollment 条件机制")
    ap.add_argument("--save-all-epochs", action="store_true",
                    help="每个 epoch 都保存 checkpoint (<ckpt-stem>_epNN.pt), "
                         "供 F1-lambda*FAR 等准则事后选模")
    ap.add_argument("--feats-subdir", default="feats",
                    help="特征子目录 (ema 微调用 feats_ema)")
    ap.add_argument("--init-from", default=None,
                    help="从已有 checkpoint 初始化 (微调)")
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os_cpu() or 8)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = MixtureDataset(args.train_dir, args.max_train or None,
                              feats_subdir=args.feats_subdir)
    val_ds = MixtureDataset(args.val_dir, feats_subdir=args.feats_subdir)
    print(f"train {len(train_ds)} 条, val {len(val_ds)} 条")
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=collate, num_workers=args.workers,
                          persistent_workers=args.workers > 0)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        collate_fn=collate, num_workers=0)

    model = MODELS[args.cond]().to(DEVICE)
    if args.init_from:
        _st = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(_st["model"])
        print(f"初始化自 {args.init_from} (ep{_st.get('epoch')})")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")
    class_weight = torch.tensor([1.0, args.weight1, args.target_weight])
    crit = nn.CrossEntropyLoss(weight=class_weight, ignore_index=-100)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    log = {"config": vars(args), "n_params": n_params, "epochs": []}
    best_f1 = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for x, e, y in train_ld:
            logits = model(x, e)
            loss = crit(logits.reshape(-1, NUM_CLASSES), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * (y != -100).sum().item()
            n += (y != -100).sum().item()
        train_loss = tot / max(n, 1)
        vm = evaluate(model, val_ld, class_weight)
        el = time.time() - t0
        entry = {"epoch": ep, "train_loss": round(train_loss, 5),
                 "elapsed_s": round(el, 1),
                 **{f"val_{k}": round(v, 5) for k, v in vm.items()}}
        log["epochs"].append(entry)
        print(f"epoch {ep:02d} ({el:.0f}s) train_loss {train_loss:.4f} "
              f"val_loss {vm['loss']:.4f} val_f1 {vm['target_f1']:.4f} "
              f"val_rec {vm['target_recall']:.4f} val_far {vm['far']:.4f}")
        if vm["target_f1"] > best_f1:
            best_f1 = vm["target_f1"]
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val": vm, "n_params": n_params,
                        "cond": args.cond}, out_dir / args.ckpt_name)
        if args.save_all_epochs:
            stem = args.ckpt_name.rsplit(".", 1)[0]
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val": vm, "n_params": n_params,
                        "cond": args.cond},
                       out_dir / f"{stem}_ep{ep:02d}.pt")
        with open(out_dir / args.log_name, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=1)
    print(f"完成, 最优 val F1 {best_f1:.4f}, checkpoint: {out_dir / args.ckpt_name}")
    return 0


def os_cpu():
    import os
    return os.cpu_count()


if __name__ == "__main__":
    sys.exit(main())
