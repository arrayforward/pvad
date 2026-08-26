导出 models/pvad/pvad.onnx 并用 onnxruntime 验证与 torch 输出一致。

输入约定:
  feats: float32 [B, T, 80]  — 与 src/fbank.cpp 相同参数的 log-mel,
                               per-utterance per-bin 均值归一化
  emb:   float32 [B, 192]    — CAM++ (models/campplus.onnx) enrollment
                               embedding, L2 归一化
输出:
  logits: float32 [B, T, 3]  — 0=静音 1=非目标语音 2=目标语音 (未 softmax)

## v3 说明 (models/pvad/pvad_v3.onnx)
- 架构: FiLM 条件 (emb 192 -> 每层 GRU 输入的 gamma/beta 逐维调制), 260,387 参数
  外部接口与 v1/v2 完全相同: feats [B,T,80] + emb [B,192] -> logits [B,T,3]
- 数据: data/mixtures_v3 (v1 干净 50% + v2 增广 50% + 700/90/90 双干扰硬负例,
  50% 样本 enrollment 经 RIR 增广), checkpoint models/pvad/best_v3.pt (ep4, 15 epoch)
- 指标 (v1 test 干净 / v2 test 增广): 帧 FAR 0.276 / 0.526, recall 0.965 / 0.958,
  端到端误打断 7.0% / 21.6% (+5 帧中值滤波: 6.6% / 20.4%)
- 中值滤波 (5 帧) 在 v3 上收益仍边际 (~1pp), C++ 端可选
