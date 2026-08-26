导出 models/pvad/pvad.onnx 并用 onnxruntime 验证与 torch 输出一致。

输入约定:
  feats: float32 [B, T, 80]  — 与 src/fbank.cpp 相同参数的 log-mel,
                               per-utterance per-bin 均值归一化
  emb:   float32 [B, 192]    — CAM++ (models/campplus.onnx) enrollment
                               embedding, L2 归一化
输出:
  logits: float32 [B, T, 3]  — 0=静音 1=非目标语音 2=目标语音 (未 softmax)
