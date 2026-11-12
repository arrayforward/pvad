# -*- coding: utf-8 -*-
"""导出 tse pvad_direct_v3 的句级二级复核头（不做量化）。"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DV = Path(__file__).resolve().parent.parent
TSE_CODE = Path(r"D:/ai-app/tse-gitee/code")
sys.path.insert(0, str(TSE_CODE))
sys.path.insert(0, str(TSE_CODE / "scripts"))
from train_multiscale_confidence_pvad import (  # noqa: E402
    ConfidenceProjection, confidence_score, score)


class DirectGateExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, windows, reference):
        identity = score(self.model, windows, reference)
        confidence = torch.sigmoid(confidence_score(self.model, windows, reference))
        return identity.unsqueeze(0), confidence.unsqueeze(0)


def main():
    torch.set_num_threads(4)
    st = torch.load(TSE_CODE / "models/pvad_direct_v3/best.pt",
                    map_location="cpu", weights_only=False)
    model = ConfidenceProjection().eval()
    model.load_state_dict(st["model"])
    wrapper = DirectGateExport(model).eval()
    windows = torch.randn(5, 192)
    reference = torch.randn(192)
    out = DV / "models/tse_e16/direct_v3.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        expected = wrapper(windows, reference)
        torch.onnx.export(
            wrapper, (windows, reference), str(out),
            input_names=["windows", "reference"],
            output_names=["identity", "confidence"],
            dynamic_axes={"windows": {0: "n_windows"}}, opset_version=17,
            do_constant_folding=True)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    actual = sess.run(None, {"windows": windows.numpy(), "reference": reference.numpy()})
    for name, a, b in zip(("identity", "confidence"), expected, actual):
        diff = float(np.max(np.abs(a.numpy() - b)))
        print(f"{name}: torch={a.numpy()} onnx={b} max_diff={diff:.3g}")
        assert diff < 1e-5
    print(f"threshold={st['threshold']} (demo 二级高召回复核使用 0.0), "
          f"uncertain={st.get('uncertain_threshold', 0.3)}")
    print(f"saved {out} ({out.stat().st_size} bytes), FP32 no quantization")


if __name__ == "__main__":
    main()
