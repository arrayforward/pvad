# pvad_v6.onnx / pvad_v6_stream.onnx — per-frame CMVN 版（实验性，未过验收线）

v6 = v5 架构（film_attn_cos，cos 注意力 + FiLM，400,516 参数）+ **per-frame CMVN**
（每帧 80 维独立减均值，无跨帧统计），从头训练 15 epoch，checkpoint
`models/pvad/best_v6.pt`（= best_v6_ep13.pt，val 两条件 e2e 增广最优点）。

## 接口（与 v5 相同）

- 离线 `pvad_v6.onnx`：`feats [B,T,80]` + `enroll_tokens [B,N,192]` + `enroll_mask [B,N]` → `logits [B,T,3]`
- 流式 `pvad_v6_stream.onnx`：另加 `h0 [2,B,128]` → 另出 `hN`

## CMVN 调用方责任（最简单的一版）

`feat_t = x_t - mean(x_t)`（逐帧 80 维减均值）。**无状态、无 warm-up、
无 EMA 参数**——训练/离线/流式天然同分布。fbank 参数同 `src/fbank.cpp`。

## 验证结果（如实）

- ONNX 对齐：离线 6.79e-06；流式 chunk=1/5/50 与 torch 整段 9.54e-06 ✓
- **流式 ≡ 离线：两条件逐数字完全相同**（分布一致性目标达成，0.0pp 差距）
- 但绝对水平未过验收线（干净 ≥91.5 / 增广 ≥87）：

| 模型 | 离线干净 | 离线增广 | 流式干净 | 流式增广 |
|---|---|---|---|---|
| **v6 ep13（本模型）** | 89.4% | 79.4% | **90.5%** | **77.5%** |
| v6 ep10（干净偏置点, 备选 ckpt） | 92.2% | 75.6% | — | — |
| 参照 v5s_stream | — | — | 92.0% | 85.0% |
| 参照 v4_stream | — | — | 94.5% | 82.5% |

## 结论与部署建议

per-frame CMVN 证明了"零漂移零 warm-up"可行，但逐帧减均值抹掉了段级
能量动态——模型区分目标/干扰所依赖的关键统计线索，增广误打断显著回升
（0.17-0.24）。**不建议替换 v4_stream（实时）或 pvad_v5.onnx（离线批量）**。
本模型仅作实验存档；ep10（best_v6_ep10.pt）为干净偏置备选。
