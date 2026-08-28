# pvad_v5_stream.onnx — v5 流式版（state 外置，EMA-CMVN 微调）

与 pvad_v5.onnx 同架构（film_attn，390,147 参数），GRU 隐状态外置，
经 EMA-CMVN 特征微调（best_v5s.pt，自 best_v5fa.pt 4 epoch lr 1e-4）。

## 输入 / 输出

| 名称 | shape | dtype |
|---|---|---|
| `feats_chunk` | [B, t, 80] | float32（CMVN 由调用方负责：EMA α=0.02 因果滑动均值） |
| `enroll_tokens` | [B, N, 192] | float32 |
| `enroll_mask` | [B, N] | bool |
| `h0` | [2, B, 128] | float32，会话起始全 0 |
| `logits` | [B, t, 3] | float32 |
| `hN` | [2, B, 128] | float32 |

## 验证结果（如实）

- chunk 对齐：chunk=1/5/50 vs torch 整段最大误差 5.19e-06 ✓
- 流式 e2e（EMA α=0.02, chunk=5, 各 200 条）：
  - v1 干净 **91.5%**（整段基线 93.5%，-2.0pp，踩线）
  - v2 增广 **71.0%**（整段基线 88.0%，**-17pp，不达标**：误打断 12%→29%）

## 结论与建议

v5 的注意力条件路径对 EMA-CMVN 统计漂移远比 v4 的 FiLM 敏感
（v4_stream 同样流程只退化 0.5pp）。**实时生产暂不建议替换 pvad_v4_stream.onnx**；
若要用 v5 流式，需要更长的 EMA 微调（>4 epoch）或注意力输入的归一化改造，
属后续工作。离线/批量场景用 pvad_v5.onnx 无此问题。
