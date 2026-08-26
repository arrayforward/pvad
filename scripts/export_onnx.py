# -*- coding: utf-8 -*-
"""导出 models/pvad/pvad.onnx 并用 onnxruntime 验证与 torch 输出一致。

输入约定:
  feats: float32 [B, T, 80]  — 与 src/fbank.cpp 相同参数的 log-mel,
                               per-utterance per-bin 均值归一化
  emb:   float32 [B, 192]    — CAM++ (models/campplus.onnx) enrollment
                               embedding, L2 归一化
输出:
  logits: float32 [B, T, 3]  — 0=静音 1=非目标语音 2=目标语音 (未 softmax)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvad_common import ROOT  # noqa: E402
from train_pvad import MODELS, FEAT_DIM, EMB_DIM  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "models" / "pvad" / "best.pt"))
    ap.add_argument("--out", default=str(ROOT / "models" / "pvad" / "pvad.onnx"))
    args = ap.parse_args()
    out = Path(args.out)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cond = state.get("cond", "concat")
    model = MODELS[cond]()
    model.load_state_dict(state["model"])
    model.eval()
    print(f"cond={cond}, epoch={state.get('epoch')}, "
          f"params={state.get('n_params'):,}")

    B, T = 2, 137
    feats = torch.randn(B, T, FEAT_DIM)
    emb = torch.randn(B, EMB_DIM)
    torch.onnx.export(
        model, (feats, emb), str(out),
        input_names=["feats", "emb"], output_names=["logits"],
        dynamic_axes={"feats": {0: "batch", 1: "time"},
                      "emb": {0: "batch"},
                      "logits": {0: "batch", 1: "time"}},
        opset_version=17)
    print(f"导出 {out}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(feats, emb).numpy()
    got = sess.run(None, {"feats": feats.numpy(), "emb": emb.numpy()})[0]
    diff = np.abs(ref - got).max()
    print(f"onnxruntime vs torch 最大绝对误差: {diff:.2e} "
          f"({'OK' if diff < 1e-4 else 'MISMATCH'})")
    # 输入约定说明
    with open(out.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(__doc__)
    return 0 if diff < 1e-4 else 1


if __name__ == "__main__":
    sys.exit(main())
