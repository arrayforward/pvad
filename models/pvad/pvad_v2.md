导出 models/pvad/pvad.onnx 并用 onnxruntime 验证与 torch 输出一致。

输入约定:
  feats: float32 [B, T, 80]  — 与 src/fbank.cpp 相同参数的 log-mel,
                               per-utterance per-bin 均值归一化
  emb:   float32 [B, 192]    — CAM++ (models/campplus.onnx) enrollment
                               embedding, L2 归一化
输出:
  logits: float32 [B, T, 3]  — 0=静音 1=非目标语音 2=目标语音 (未 softmax)

## v2 说明 (models/pvad/pvad_v2.onnx)
- 训练数据: data/mixtures_v2 (RIR 70% + MUSAN 噪声 60% 增广, 含 21% target-absent 负样本)
- 12 epoch, 损失权重 [1,2,3], checkpoint models/pvad/best_v2.pt (ep12)
- 与 v1 (pvad.onnx) 输入输出约定完全相同, 可直接替换加载
- 帧输出中值滤波 (5 帧) 在 v2 test 上仅带来 ~0.4pp 的误打断改善 (18.0%->17.6%),
  收益边际; 若 C++ 端实现成本低可加, 否则不必
