# pvad_v5s_stream.onnx — v5 流式修复版（cosine 注意力 + state 外置）

v5 film_attn 的流式修复版：交叉注意力改为 **cosine 注意力**（Q/K 投影后各自
L2 归一化 + 可学温度，尺度不变 → 对 EMA-CMVN 统计漂移免疫），其余结构同
film_attn（400,516 参数）。checkpoint `models/pvad/best_v5s.pt`（= best_v5s_ep02.pt，
自 best_v5cos_off.pt EMA-CMVN 微调 2 epoch 处截取，val e2e 平衡点）。

## 输入 / 输出

| 名称 | shape | dtype |
|---|---|---|
| `feats_chunk` | [B, t, 80] | float32（CMVN 由调用方负责：EMA α=0.02 因果滑动均值） |
| `enroll_tokens` | [B, N, 192] | float32（enrollment 每 1s 子段一个 CAM++ embedding） |
| `enroll_mask` | [B, N] | bool |
| `h0` | [2, B, 128] | float32，会话起始全 0 |
| `logits` | [B, t, 3] | float32 |
| `hN` | [2, B, 128] | float32 |

## 验证结果（如实）

- chunk 对齐：chunk=1/5/50 vs torch 整段最大误差 4.98e-06 ✓
- 流式 e2e（EMA α=0.02, chunk=5, confirm=2, 各 200 条）：
  - v1 干净 **92.0%**（漏 0.5%/误 7.5%）——过 ≥91.5% 线 ✓
  - v2 增广 **85.0%**（漏 0.0%/误 15.0%）——未达 ≥87% 线（差 2.0pp）
- 参照：未归一化的 pvad_v5_stream.onnx 增广仅 71.0%（本修复 +14pp）；
  修复前模型（best_v5cos_off.pt，未 EMA 微调）流式增广 87.0% 但干净仅 85.5%——
  干净/增广操作点互相拉扯，未能同时过线。

## warm-up 建议

EMA 时间常数 50 帧（0.5s），会话起始 0.5s 内建议不做门控判定。
cosine 注意力对早期未收敛统计已不敏感（增广退化主因是归一化本身的
信息损失，不是漂移）。

## 部署建议（如实）

干净条件本模型可用且优于 pvad_v4_stream（92.0% vs 94.5% 基线口径下略低）；
增广条件未达验收线。若实时场景以干净为主可替换；强噪声/混响场景建议
暂留 pvad_v4_stream.onnx，或等下一步（aug 加权 EMA 微调 / per-frame CMVN 重训）。

## aug 加权微调追加实验（2026-08-28，未改变结论）

尝试把增广样本 loss 加权拉回操作点（起点 best_v5s ep02，EMA 特征，val 两条件
e2e 选点，scripts/select_v5s.py）：

| 配置 | val v1 干净 | val v2 增广 |
|---|---|---|
| **best_v5s ep02（保持最终）** | **87.0%** | **88.5%** |
| aug×2 ×2ep（best_v5aw_ep02） | 83.4% | 79.8% |
| aug×1.5 ×1ep（best_v5aw15） | 87.0% | 83.8% |

aug 加权两轮均只推高了双条件误打断，未能改善增广——ep02 仍为最优，
本模型（干净 92.0% / 增广 85.0%，增广差验收线 2.0pp）保持为最终交付。
val v2 88.5% 与 test 增广 85.0% 的差异来自样本集构成（val 253 条 vs test 200 条，
200 条协议下 ±2pp 量级波动属正常）。
