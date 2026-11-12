# -*- coding: utf-8 -*-
"""TSE (目标说话人提取) SpeakerBeam-mask 模型训练。

模型: logmag [B,T,257] -> Linear(257->384) -> FiLM(emb) -> GRU(384)
      -> FiLM(emb) -> GRU(384) -> Linear(384->257) sigmoid ratio mask
      emb(192) -> Linear 生成每层 GRU 输入的 (gamma,beta) (PVAD v3+ 验证机制)
      参数量 ~1.94M (预算 2-5M)
STFT/ISTFT 在模型外 (python/C++ 侧, 与 PVAD fbank 同思路, 不进 ONNX)。

损失: 压缩域幅度 L1 (|S|^0.3) + 非对称项: est<tgt (过抑制) 的 bin 权重 x2.5。

用法:
  python scripts/train_tse.py --epochs 20 --batch 16 --lr 1e-3
  python scripts/train_tse.py --init-from models/tse/best_tse.pt --epochs 5
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
from pvad_common import ROOT, DATA_ROOT  # noqa: E402

SR = 16000
N_FFT = 512
HOP = 160
WIN = 400
N_FREQ = N_FFT // 2 + 1  # 257
EMB_DIM = 192
HIDDEN = 384
WINDOW = torch.hamming_window(WIN)


def stft_mag(pcm_t):
    """[n] float32 -> mag [T, 257] (center=True, hamming)。"""
    spec = torch.stft(pcm_t, N_FFT, HOP, WIN, window=WINDOW.to(pcm_t.device),
                      center=True, return_complex=True)
    return spec.abs().transpose(0, 1)  # [T, 257]


class TseMaskModel(nn.Module):
    def __init__(self, n_freq=N_FREQ, emb_dim=EMB_DIM, hidden=HIDDEN):
        super().__init__()
        self.in_proj = nn.Linear(n_freq, hidden)
        self.film_in = nn.Linear(emb_dim, 2 * hidden)
        self.gru1 = nn.GRU(hidden, hidden, batch_first=True)
        self.film_h = nn.Linear(emb_dim, 2 * hidden)
        self.gru2 = nn.GRU(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_freq)

    def forward(self, logmag, emb):
        h = self.in_proj(logmag)
        gb1 = self.film_in(emb).unsqueeze(1)
        h = h * (1.0 + gb1[..., :h.shape[-1]]) + gb1[..., h.shape[-1]:]
        h, _ = self.gru1(h)
        gb2 = self.film_h(emb).unsqueeze(1)
        h = h * (1.0 + gb2[..., :h.shape[-1]]) + gb2[..., h.shape[-1]:]
        h, _ = self.gru2(h)
        return torch.sigmoid(self.fc(h))


class TseDataset(Dataset):
    def __init__(self, split_dir, max_n=None):
        d = Path(split_dir)
        self.dir = d
        self.recs = []
        with open(d / "meta.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.recs.append(json.loads(line))
        if max_n:
            import random
            random.Random(1234).shuffle(self.recs)
            self.recs = self.recs[:max_n]
        self.recs = [r for r in self.recs
                     if (d / "emb" / f"{r['id']}.npy").exists()]

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        import soundfile as sf
        r = self.recs[i]
        mix, _ = sf.read(str(self.dir / f"{r['id']}_mix.wav"), dtype="float32")
        tgt, _ = sf.read(str(self.dir / f"{r['id']}_target.wav"), dtype="float32")
        emb = np.load(self.dir / "emb" / f"{r['id']}.npy")
        mix_t = torch.from_numpy(mix)
        tgt_t = torch.from_numpy(tgt)
        logmag_mix = torch.log(stft_mag(mix_t).clamp_min(1e-5))
        mag_tgt = stft_mag(tgt_t)
        return (logmag_mix, mag_tgt,
                torch.from_numpy(emb), r["augmented"], r["negative"])


def collate(batch):
    lms, mts, es, _, _ = zip(*batch)
    B = len(batch)
    T = max(x.shape[0] for x in lms)
    lm = torch.zeros(B, T, N_FREQ)
    mt = torch.zeros(B, T, N_FREQ)
    emb = torch.stack(es)
    pad_mask = torch.ones(B, T, dtype=torch.bool)
    for i, (a, b) in enumerate(zip(lms, mts)):
        t = a.shape[0]
        lm[i, :t] = a
        mt[i, :t] = b
        pad_mask[i, :t] = False
    return lm, mt, emb, pad_mask


def si_snr(est, ref, eps=1e-8):
    """[n] 波形 SI-SNR (dB)。"""
    est = est - est.mean()
    ref = ref - ref.mean()
    proj = (est * ref).sum() / (ref * ref).sum().clamp(min=eps) * ref
    noise = est - proj
    return 10 * torch.log10((proj * proj).sum().clamp(min=eps)
                            / (noise * noise).sum().clamp(min=eps))


@torch.no_grad()
def evaluate(model, loader, n_eval=200):
    model.eval()
    si_before, si_after, os_rates = [], [], []
    for k, (lm, mt, emb, pm) in enumerate(loader):
        mask = model(lm, emb)
        mix_mag = torch.exp(lm) - 1e-5
        est_mag = mask * mix_mag
        for i in range(lm.shape[0]):
            t = (~pm[i]).sum().item()
            # 压缩域指标 (波形级 SI-SNR 在独立评估脚本做, 这里快速用谱域近似)
            est_c = est_mag[i, :t] ** 0.3
            tgt_c = mt[i, :t] ** 0.3
            si_after.append(-(est_c - tgt_c).abs().mean().item())
            base_c = mix_mag[i, :t] ** 0.3
            si_before.append(-(base_c - tgt_c).abs().mean().item())
            # 过抑制: est 幅度能量 vs tgt 幅度能量
            e_est = (est_mag[i, :t] ** 2).sum()
            e_tgt = (mt[i, :t] ** 2).sum().clamp(min=1e-8)
            os_rates.append(float(10 * torch.log10(e_est.clamp(min=1e-8) / e_tgt) < -6))
        if (k + 1) * lm.shape[0] >= n_eval:
            break
    si_imp = np.mean(si_after) - np.mean(si_before)
    return {"si_imp_spec": float(si_imp), "os_rate": float(np.mean(os_rates)),
            "spec_err": float(np.mean(si_after))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default=str(DATA_ROOT / "tse_mix" / "train"))
    ap.add_argument("--val-dir", default=str(DATA_ROOT / "tse_mix" / "val"))
    ap.add_argument("--out-dir", default=str(ROOT / "models" / "tse"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--os-weight", type=float, default=2.5)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--ckpt-name", default="best_tse.pt")
    ap.add_argument("--log-name", default="train_log_tse.json")
    ap.add_argument("--save-all-epochs", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = TseDataset(args.train_dir, args.max_train or None)
    val_ds = TseDataset(args.val_dir)
    print(f"train {len(train_ds)} 条, val {len(val_ds)} 条")
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=collate, num_workers=args.workers,
                          persistent_workers=args.workers > 0)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        collate_fn=collate, num_workers=0)

    model = TseMaskModel()
    if args.init_from:
        st = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        print(f"初始化自 {args.init_from} (ep{st.get('epoch')})")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    log = {"config": vars(args), "n_params": n_params, "epochs": []}
    best_score = -1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for lm, mt, emb, pm in train_ld:
            mask = model(lm, emb)
            mix_mag = torch.exp(lm) - 1e-5
            est_mag = mask * mix_mag
            est_c = est_mag ** 0.3
            tgt_c = mt ** 0.3
            diff = est_c - tgt_c
            # 非对称: est<tgt (过抑制) 权重 x os_weight; padding 帧不计
            w = torch.where(diff < 0, args.os_weight, 1.0)
            w = w * (~pm).float().unsqueeze(-1)
            loss = (w * diff.abs()).sum() / w.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        vm = evaluate(model, val_ld)
        el = time.time() - t0
        score = vm["si_imp_spec"] - 2.0 * vm["os_rate"]  # 复合: 改善 - 过抑制惩罚
        entry = {"epoch": ep, "train_loss": round(tot / max(n, 1), 5),
                 "elapsed_s": round(el, 1), "val_score": round(score, 5),
                 **{f"val_{k}": round(v, 5) for k, v in vm.items()}}
        log["epochs"].append(entry)
        print(f"epoch {ep:02d} ({el:.0f}s) loss {entry['train_loss']:.4f} "
              f"val_si_imp {vm['si_imp_spec']:+.4f} val_os {vm['os_rate']:.3f} "
              f"score {score:+.4f}")
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "epoch": ep, "val": vm,
                        "n_params": n_params}, out_dir / args.ckpt_name)
        if args.save_all_epochs:
            stem = args.ckpt_name.rsplit(".", 1)[0]
            torch.save({"model": model.state_dict(), "epoch": ep, "val": vm,
                        "n_params": n_params}, out_dir / f"{stem}_ep{ep:02d}.pt")
        with open(out_dir / args.log_name, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=1)
    print(f"完成, 最优 score {best_score:+.4f}, ckpt {out_dir / args.ckpt_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
