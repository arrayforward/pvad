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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "models" / "pvad" / "best_v4.pt"))
    ap.add_argument("--out", default=str(ROOT / "models" / "pvad" / "pvad_v4_stream.onnx"))
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cond = state.get("cond", "concat")
    assert cond == "film", f"流式导出仅支持 film 架构, 当前 {cond}"
    model = MODELS[cond]()
    model.load_state_dict(state["model"])
    model.eval()
    wrapped = StreamWrapper(model)
    wrapped.eval()
    print(f"ckpt ep{state.get('epoch')}, params={state.get('n_params'):,}")

    out = Path(args.out)
    B, T = 2, 5
    feats = torch.randn(B, T, FEAT_DIM)
    emb = torch.randn(B, EMB_DIM)
    h0 = torch.zeros(2, B, HIDDEN)
    torch.onnx.export(
        wrapped, (feats, emb, h0), str(out),
        input_names=["feats_chunk", "emb", "h0"], output_names=["logits", "hN"],
        dynamic_axes={"feats_chunk": {0: "batch", 1: "time"},
                      "emb": {0: "batch"}, "h0": {1: "batch"},
                      "logits": {0: "batch", 1: "time"}, "hN": {1: "batch"}},
        opset_version=17)
    print(f"导出 {out}")

    # ---- 对齐验证: 整段一次性 vs chunk=1/5/50 分段 ----
    import onnxruntime as ort
    sess_full = ort.InferenceSession(str(ROOT / "models" / "pvad" / "pvad_v4.onnx"),
                                     providers=["CPUExecutionProvider"])
    sess_stream = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 137, FEAT_DIM)).astype(np.float32)
    e = rng.standard_normal((1, EMB_DIM)).astype(np.float32)
    ref = sess_full.run(None, {"feats": x, "emb": e})[0]
    worst = 0.0
    for chunk in (1, 5, 50):
        h = np.zeros((2, 1, HIDDEN), dtype=np.float32)
        outs = []
        for s in range(0, x.shape[1], chunk):
            lg, h = sess_stream.run(
                None, {"feats_chunk": x[:, s:s + chunk], "emb": e, "h0": h})
            outs.append(lg)
        got = np.concatenate(outs, axis=1)
        d = float(np.abs(ref - got).max())
        worst = max(worst, d)
        print(f"chunk={chunk:>2}: 与整段 ONNX 最大误差 {d:.2e}")
    ok = worst < 1e-5
    print(f"总体 {'OK' if ok else 'MISMATCH'} (<1e-5)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
