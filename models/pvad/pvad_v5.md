# pvad_v5.onnx — enrollment 多帧表示 + 交叉注意力 + FiLM (film_attn)

v5 = enrollment 音频按 1s 切子段（整 1s 切分，不足 1s 尾段丢弃；enrollment ≥3s
故 N≥3），每段过 CAM++ 得 192 维向量 → token 序列 [N,192]；
帧特征投影作 query、tokens 作 K/V 的手写 2 头交叉注意力（残差融合），
FiLM 用 mask 均值池化的 tokens 继续逐层调制，GRU 2×128。**390,147 参数**。
checkpoint `models/pvad/best_v5fa.pt`（best_v5fa_ep08.pt，val e2e 选模，
自 best_v4.pt 的 GRU/输出层初始化训练 10 epoch）。

## 输入

| 名称 | shape | dtype | 说明 |
|---|---|---|---|
| `feats` | [B, T, 80] | float32 | log-mel fbank（同 `src/fbank.cpp`）+ per-bin 均值归一化 |
| `enroll_tokens` | [B, N, 192] | float32 | enrollment 每 1s 子段一个 CAM++ embedding，各 L2 归一化 |
| `enroll_mask` | [B, N] | bool | True = padding 位（N 不足批次最大时补 True） |

## 输出

`logits` [B, T, 3] float32 — 0=静音 1=非目标 2=目标（未 softmax）

## 指标（python 口径，各 500 条，confirm=2、±20 帧）

| 条件 | 漏打断 | 误打断 | 正确率 | 对比 v4 |
|---|---|---|---|---|
| v1 干净 | 1.4% | 5.6% | **93.0%** | 94.0%（-1.0pp，验收线内） |
| v2 增广 | 0.6% | **10.0%** | **89.4%** | 83.0%（**+6.4pp**，误打断 16.4%→10.0%） |

帧级（ALL）：干净 FAR 0.225/recall 0.891；增广 FAR 0.435/recall 0.878。
onnxruntime vs torch 最大误差 5.25e-06。

## 部署注意

- 离线/批量路径可直接替换 pvad_v4.onnx（注意输入从单个 emb 向量变为
  tokens+mask，C++ 侧 enrollment 需按 1s 子段多次过 CAM++）
- 实时流式路径见 pvad_v5_stream.md——**v5 流式版增广条件不达标，
  实时生产建议暂留 pvad_v4_stream.onnx**
