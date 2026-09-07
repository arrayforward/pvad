导出 models/pvad/pvad.onnx 并用 onnxruntime 验证与 torch 输出一致。

输入约定:
  feats: float32 [B, T, 80]  — 与 src/fbank.cpp 相同参数的 log-mel,
                               per-utterance per-bin 均值归一化
  emb:   float32 [B, 192]    — CAM++ (models/campplus.onnx) enrollment
                               embedding, L2 归一化
输出:
  logits: float32 [B, T, 3]  — 0=静音 1=非目标语音 2=目标语音 (未 softmax)

## v7b 最终数值（ep01，长短批内 50/50 联合微调，自 best_v4s）

- 标准 e2e（EMA 流式，各 200 条）：干净 **96.5%**（漏 0.0/误 3.5%）、
  增广 79.5%（漏 0.0/误 20.5%）——干净超 v4s(94.5)，增广差线 1.0pp
- 90s 剧本：目标 4/4、噪声误触发 1、非目标误触发 3（confirm=2/4 均 3）
- 合成长流 300 条：目标段召回 75.3%（v4s 19%）、纯负零触发 20%（v4s 2.5%）
- 冷启动前 50 帧 meanP 0.010（v4s 0.012），maxP 0.014（v4s 0.058）
- ONNX 对齐：离线 2.62e-06 / 流式 3.81e-06
- **验收未全过（①非目标 3>1；②增广 79.5<80.5），不替换生产**；
  建议维持 pvad_v4_stream.onnx + 会话策略缓解，v7b 存档供后续混合比例探索
