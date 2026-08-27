# pvad_v4_stream.onnx — 流式 (chunked state 复用) PVAD

与 `pvad_v4.onnx` 同架构（FiLM 条件，260,387 参数），GRU 隐状态外置，
供 C++ 实时管线按 chunk 增量推理，避免每帧全段重算。
**注意：本模型在 EMA-CMVN 特征上微调过（best_v4s.pt，4 epoch lr 1e-4，
自 best_v4.pt 初始化），用于流式路径；离线整段路径仍用 pvad_v4.onnx。**

## 输入

| 名称 | shape | dtype | 说明 |
|---|---|---|---|
| `feats_chunk` | [B, t, 80] | float32 | 当前 chunk 的 log-mel fbank（参数同 `src/fbank.cpp`）。**CMVN 由调用方负责，不进 ONNX** |
| `emb` | [B, 192] | float32 | CAM++ enrollment embedding，L2 归一化 |
| `h0` | [2, B, 128] | float32 | 两层 GRU 隐状态，**会话起始为全 0** |

## 输出

| 名称 | shape | dtype | 说明 |
|---|---|---|---|
| `logits` | [B, t, 3] | float32 | 0=静音 1=非目标 2=目标（未 softmax） |
| `hN` | [2, B, 128] | float32 | 下一 chunk 回传为 `h0` |

## CMVN 约定（关键）

训练/微调使用的因果 CMVN：**指数滑动均值，α=0.02**（时间常数 50 帧 = 0.5s），
从流起始累积：

```
m_0 = 0
m_t = 0.98 * m_{t-1} + 0.02 * x_t      # 逐 bin
feat_t = x_t - m_t
```

调用方每帧 O(80) 更新即可。**不要用整段均值 / 滑窗均值 / running 均值**
（实测 e2e 大幅退化，见 TRAINING.md 第 10 节）。
enrollment embedding 的计算不变（enrollment 是离线整段音频，仍用整段 CMVN + CAM++）。

## warm-up

EMA 均值约 0.5s（50 帧）收敛。微调后模型对早期未收敛统计鲁棒，
实测 3-8s 测试片段（语音从片段起始就出现的最坏情况）流式 e2e 与整段基线
仅差 0.5pp。真实会话开头若有 ≥0.5s 任意音频（静音/唤醒词），偏差可忽略。

## 对齐验证（python，scripts/export_stream_onnx.py / eval_stream.py）

- 同一整段输入，chunk=1/5/50 分段喂 + state 串接 vs torch 整段：
  逐帧 logits 最大误差 **2.62e-06**（数学等价，数值噪声级）
- 流式 e2e（EMA CMVN, chunk=5, confirm=2, ±20 帧, 各 200 条）：
  v1 干净 **94.5%**（整段基线 95.0%，-0.5pp），
  v2 增广 **82.5%**（整段基线 83.0%，-0.5pp）
