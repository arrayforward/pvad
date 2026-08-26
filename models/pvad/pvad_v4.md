导出 models/pvad/pvad.onnx 并用 onnxruntime 验证与 torch 输出一致。

输入约定:
  feats: float32 [B, T, 80]  — 与 src/fbank.cpp 相同参数的 log-mel,
                               per-utterance per-bin 均值归一化
  emb:   float32 [B, 192]    — CAM++ (models/campplus.onnx) enrollment
                               embedding, L2 归一化
输出:
  logits: float32 [B, T, 3]  — 0=静音 1=非目标语音 2=目标语音 (未 softmax)

## v4 说明 (models/pvad/pvad_v4.onnx)
- 架构: FiLM 条件 (同 v3), 260,387 参数, 接口 feats [B,T,80] + emb [B,192] -> logits [B,T,3]
- 数据: data/mixtures_v4 (v3 基础 + 2000 条音色相近易混淆负样本, 干扰人为目标说话人
  CAM++ 质心 top5 近邻; 负样本占比 25%: 易混淆 2000 + 双干扰 700 + 随机 2116)
- 选模: 15 epoch 全存 checkpoint; F1-λ·FAR (λ=1/2) 均指向 ep14 但其 val e2e 更差,
  最终按 val e2e 选 ep11 (best_v4_ep11.pt)
- python 评估 (v1 干净 / v2 增广): 帧 FAR 0.210 / 0.446, recall 0.940 / 0.926;
  e2e 误打断 5.0% / 16.4%, 正确率 94.0% / 83.0% (两条件均为历代最优)
