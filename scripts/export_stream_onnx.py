# -*- coding: utf-8 -*-
"""导出流式 (chunked state 复用) PVAD ONNX: models/pvad/pvad_v4_stream.onnx。

输入:
  feats_chunk: float32 [B, t, 80]  — 当前 chunk 的 fbank (CMVN 由调用方负责, 不进 ONNX)
  emb:         float32 [B, 192]    — CAM++ enrollment embedding, L2 归一化
  h0:          float32 [2, B, 128] — GRU 隐状态 (两层), 初始为全 0
输出:
  logits:      float32 [B, t, 3]
  hN:          float32 [2, B, 128] — 供下一 chunk 回传
与整段模型数学等价 (CMVN 相同时逐帧 logits 严格一致)。

用法: python scripts/export_stream_onnx.py [--ckpt models/pvad/best_v4.pt]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvad_common import ROOT  # noqa: E402
from train_pvad import MODELS, FEAT_DIM, EMB_DIM  # noqa: E402

HIDDEN = 128


class StreamWrapper(torch.nn.Module):
    """FiLM PVAD 的 state 外置封装: (feats_chunk, emb, h0) -> (logits, hN)。"""

    def __init__(self, model):
        super().__init__()
        self.film_in = model.film_in
        self.gru1 = model.gru1
        self.film_h = model.film_h
        self.gru2 = model.gru2
        self.fc = model.fc

    def forward(self, feats_chunk, emb, h0):
        fd = feats_chunk.shape[-1]
        gb1 = self.film_in(emb).unsqueeze(1)
        h = feats_chunk * (1.0 + gb1[..., :fd]) + gb1[..., fd:]
        h, h1 = self.gru1(h, h0[0:1])
        gb2 = self.film_h(emb).unsqueeze(1)
        hd = h.shape[-1]
        h = h * (1.0 + gb2[..., :hd]) + gb2[..., hd:]
        h, h2 = self.gru2(h, h0[1:2])
        return self.fc(h), torch.cat([h1, h2], dim=0)


class StreamWrapperV5(torch.nn.Module):
    """v5 film_attn 系的 state 外置封装 (兼容 ln/cos 变体):
    (feats_chunk, tokens, kv_mask, h0) -> (logits, hN)。"""

    def __init__(self, model):
        super().__init__()
        self.has_ln = hasattr(model, "ln_q")
        self.has_res = hasattr(model, "res_proj")
        if self.has_ln:
            self.ln_q = model.ln_q
        self.in_proj = model.in_proj
        if self.has_res:
            self.res_proj = model.res_proj
        self.attn = model.attn
        self.film_in = model.film_in
        self.gru1 = model.gru1
        self.film_h = model.film_h
        self.gru2 = model.gru2
        self.fc = model.fc

    def forward(self, feats_chunk, tokens, kv_mask, h0):
        q_in = self.ln_q(feats_chunk) if self.has_ln else feats_chunk
        q = self.in_proj(q_in)
        a, _ = self.attn(q, tokens, kv_mask)
        res = self.res_proj(feats_chunk) if self.has_res else q
        h = res + a
        keep = (~kv_mask).float().unsqueeze(-1)
        pooled = (tokens * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        hd = h.shape[-1]
        gb1 = self.film_in(pooled).unsqueeze(1)
        h = h * (1.0 + gb1[..., :hd]) + gb1[..., hd:]
        h, h1 = self.gru1(h, h0[0:1])
        gb2 = self.film_h(pooled).unsqueeze(1)
        h = h * (1.0 + gb2[..., :hd]) + gb2[..., hd:]
        h, h2 = self.gru2(h, h0[1:2])
        return self.fc(h), torch.cat([h1, h2], dim=0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "models" / "pvad" / "best_v4.pt"))
    ap.add_argument("--out", default=str(ROOT / "models" / "pvad" / "pvad_v4_stream.onnx"))
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cond = state.get("cond", "concat")
    assert cond in ("film", "film_attn", "film_attn_ln", "film_attn_cos"), \
        f"流式导出仅支持 film/film_attn 系, 当前 {cond}"
    model = MODELS[cond]()
    model.load_state_dict(state["model"])
    model.eval()
    v5 = cond != "film"
    wrapped = StreamWrapperV5(model) if v5 else StreamWrapper(model)
    wrapped.eval()
    print(f"ckpt ep{state.get('epoch')}, params={state.get('n_params'):,}")

    out = Path(args.out)
    B, T = 2, 5
    feats = torch.randn(B, T, FEAT_DIM)
    h0 = torch.zeros(2, B, HIDDEN)
    if v5:
        from torch.export import Dim
        N = 6
        toks = torch.randn(B, N, EMB_DIM)
        kv_mask = torch.zeros(B, N, dtype=torch.bool)
        ds = {"feats_chunk": {0: Dim("batch"), 1: Dim("time")},
              "tokens": {0: Dim("batch"), 1: Dim("n_tok")},
              "kv_mask": {0: Dim("batch"), 1: Dim("n_tok")},
              "h0": {1: Dim("batch")}}
        torch.onnx.export(
            wrapped, (feats, toks, kv_mask, h0), str(out),
            input_names=["feats_chunk", "enroll_tokens", "enroll_mask", "h0"],
            output_names=["logits", "hN"],
            dynamic_shapes=ds, opset_version=17)
    else:
        emb = torch.randn(B, EMB_DIM)
        torch.onnx.export(
            wrapped, (feats, emb, h0), str(out),
            input_names=["feats_chunk", "emb", "h0"], output_names=["logits", "hN"],
            dynamic_axes={"feats_chunk": {0: "batch", 1: "time"},
                          "emb": {0: "batch"}, "h0": {1: "batch"},
                          "logits": {0: "batch", 1: "time"}, "hN": {1: "batch"}},
            opset_version=17)
    print(f"导出 {out}")

    # ---- 对齐验证: torch 整段 vs chunk=1/5/50 分段 (同一权重) ----
    import onnxruntime as ort
    sess_stream = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 137, FEAT_DIM)).astype(np.float32)
    with torch.no_grad():
        if v5:
            e_np = rng.standard_normal((1, 6, EMB_DIM)).astype(np.float32)
            m_np = np.zeros((1, 6), dtype=bool)
            ref = model(torch.from_numpy(x), torch.from_numpy(e_np),
                        torch.from_numpy(m_np)).numpy()
        else:
            e_np = rng.standard_normal((1, EMB_DIM)).astype(np.float32)
            ref = model(torch.from_numpy(x), torch.from_numpy(e_np)).numpy()
    worst = 0.0
    for chunk in (1, 5, 50):
        h = np.zeros((2, 1, HIDDEN), dtype=np.float32)
        outs = []
        for s in range(0, x.shape[1], chunk):
            if v5:
                lg, h = sess_stream.run(
                    None, {"feats_chunk": x[:, s:s + chunk],
                           "enroll_tokens": e_np, "enroll_mask": m_np, "h0": h})
            else:
                lg, h = sess_stream.run(
                    None, {"feats_chunk": x[:, s:s + chunk], "emb": e_np, "h0": h})
            outs.append(lg)
        got = np.concatenate(outs, axis=1)
        d = float(np.abs(ref - got).max())
        worst = max(worst, d)
        print(f"chunk={chunk:>2}: 与 torch 整段最大误差 {d:.2e}")
    ok = worst < 1e-5
    print(f"总体 {'OK' if ok else 'MISMATCH'} (<1e-5)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
