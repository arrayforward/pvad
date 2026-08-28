# 设计文档（DESIGN）

本文档记录 pvad-barge-in 系统的问题定义、设计决策史（含被否决方案及理由）、PVAD 模型
设计、门控状态机、C++ 工程架构与全部实测数据。使用说明见 [USAGE.md](USAGE.md)，
训练全流程见 [TRAINING.md](../TRAINING.md)。

---

## 1. 问题定义

语音交互"双讲"（double-talk / barge-in）场景：AI 正在播放 TTS，用户想直接用语音打断。
门控系统必须在**只有已注册用户 A 说话时才触发打断**，同时压制三类干扰源：

1. **TTS 自身回声**：扬声器播出的 TTS 经空气/设备回环进入麦克风。音色已知（TTS 引擎固定），
   可能与 A 同时存在（双讲）；
2. **非注册用户 B**：音色任意、事先未知；
3. **双讲混合**：A 与 TTS/B 同时发声，A 的观测被混合污染——这是最难的一条。

全部 16kHz 单声道，中文为主。系统形态：麦克风 → RNNoise 降噪（默认开）→ PVAD 帧级
目标说话人检测 → 迟滞门控 → 停止 TTS 播放（AEC 与 AS-norm 门控为保留的可选路径）。

---

## 2. 设计决策史

系统经历四代门控方案，每一代都由上一代的实测缺陷驱动。

### 2.1 为什么不用 AEC（第一代被降级为可选）

最初管线是教科书式的 `mic → AEC → VAD → 声纹打分`。AEC 用 SpeexDSP（mdf.c 源码编入），
离线模拟（mic = 语音 + 0.6×TTS，零延迟理想回环）下有效：回声段声纹相似度从 0.42 压到 0.27。

**降级理由**：实时系统中 AEC 参考信号（TTS 播放流）与采集流走不同声卡时钟域，
存在 ppm 级采样率漂移，长时间运行必然失锁；对齐稍偏，自适应滤波器会把**近端语音**
（正是要检测的 A）当作回声消掉。这是业内公认的血泪问题，原型阶段决定不引入，
改为"声纹门控自带抗回声"。SpeexDSP 代码保留为显式可选项（`--aec`，仅用于对比实验）。

### 2.2 负模板 + margin 判决（第二代）

注册时把 **TTS 音色作为负样本**存入模板（负质心），判决改为
`sA > threshold AND (sA − sNeg) > margin`（sA = 与 A 质心的余弦，sNeg = 与负质心最大余弦）。
TTS 回声段的 embedding 与 TTS 负质心高度相似，margin 被压成负值，**无需任何参考信号对齐**。
实测（SAPI 样本）：纯回声 sA≈0.33 / sNeg≈1.0，干净拒识；B 说话 sA 峰值 0.42 < 0.55。

### 2.3 AS-norm cohort 打分归一化（第三代）

问题：换设备/环境/加噪后余弦相似度**绝对值整体漂移**，固定阈值失守
（实测 A 的滑窗 sA P95 从干净 0.702 跌到 SNR10 的 0.529，跌破阈值 0.55）。

方案：test-side t-norm / AS-norm 风格——模板存 N 个独立 cohort embedding，
对滑窗 embedding x 计算 cohort 余弦的 top-K 均值 μ 和标准差 σ，
`z = (sA − μ) / max(σ, 1e-4)`，双判据 `z > 3.0 AND margin > 0.15`。
噪声同时拉低 x 对质心和对 cohort 的相似度，μ/σ 随环境自适应。实测 A 的 z P95 三档
（干净/SNR20/SNR10）稳定在 19.7–21.8，SNR10 下 INTERRUPT 从纯余弦的 2 次恢复到 6 次。

**两个教训**（详细数据见 USAGE.md 5.3）：cohort 质量是瓶颈（SAPI pitch/rate 变体
不是"不同说话人"，cos 高达 0.65–0.95，必须剔除）；z 在 impostor 上有离群值
（max 3.0–4.1，σ 极小的帧），margin 判据不可去掉。

### 2.4 PVAD 融合方案（第四代，当前默认）

AS-norm 的根本缺陷：**双讲时滑窗 embedding 是 A 与干扰的混合**，sA↓、sNeg↑，
margin 被双向压缩。端到端评估给出量化判决（500 条含双讲样本，test 池）：

| 门控 | 干净 e2e 正确率 | 增广 e2e 正确率 |
|---|---|---|
| CAM++ AS-norm | **35.2%**（漏打断 64.8%） | **26.0%**（漏打断 74.0%） |
| PVAD v3 | **92.4%** | **78.2%** |

这正是 **Personal VAD 论文的动机**：说话人确认（SV）+ VAD 的级联/组合在双讲下必然退化，
正确做法是**以注册 embedding 为条件做帧级的目标说话人语音检测**——把"这段帧是不是 A
在说话"建模为一个条件三分类问题（静音/非目标/目标）。我们自训练的 PVAD 就是该思路的
实现，AS-norm 基线保留为对照（`--gate asnorm`）。

### 2.5 v4 易混淆硬负例与 RNNoise 默认开（当前形态）

**v4 数据突破**：v3 的误打断分析显示大量假阳性来自"干扰人音色与目标相近"的样本。
v4 在 v3 数据基础上追加 **2000 条易混淆硬负例**——干扰人按目标说话人 CAM++ 质心的
top5 近邻挑选（负样本占比 25%：易混淆 2000 + 双干扰 700 + 随机 2116）。
效果：帧 FAR 干净 0.276→**0.210**、增广 0.526→**0.446**，e2e 误打断干净 7.0%→**5.0%**、
增广 21.6%→**16.4%**，两条件正确率均为历代最优（94.0%/83.0%），v4 成为默认模型
（架构与 v3 完全相同，纯数据改进）。

**RNNoise 默认开**：A/B 实测（第 7 节）显示干净条件下降噪同时改善 A（0.68→0.80）与
B（0.48→0.38）、开销仅 0.6%，而"双讲判定零改善"的最差情况也不退化，故从可选改为
默认开（`--denoise off` 可回滚）。同期教训一个：qt_demo auto-test 曾跨场景共享
Denoise 实例，重采样器/RNNoise 残态污染后续文件特征（v4 下 voice1b 0.47 漏触发），
修为每文件新实例后与 CLI 逐位一致——**有状态前件必须按文件/流隔离实例**。

---

## 3. PVAD 模型设计

### 3.1 接口与特征

- 输入 `feats [B,T,80]`：80 维 log-mel fbank（25ms 窗/10ms 移、fft512、预加重 0.97、
  对称 hamming、low 20Hz、log(max(e,1e-10))）+ **per-utterance per-bin 均值归一化**。
  python（`scripts/pvad_common.py`）与 C++（`src/fbank.cpp`）逐参数对齐，
  已用 onnxruntime 双边验证数值一致（逐帧相同）；
- 输入 `emb [B,192]`：CAM++（`models/campplus.onnx`，3D-Speaker 中文模型）
  enrollment embedding，L2 归一化；
- 输出 `logits [B,T,3]`：0 静音 / 1 非目标语音 / 2 目标语音，10ms 帧率。

### 3.2 架构与条件机制

| | v1 / v2 | v3 / v4 | **v5（离线默认）** |
|---|---|---|---|
| 条件机制 | emb broadcast 拼接到每帧（272 维输入） | **FiLM**：emb → 每层 GRU 输入的 (γ,β) 逐维调制 `h⊙(1+γ)+β` | **多帧 tokens + 交叉注意力 + FiLM** |
| enrollment 表示 | 单向量（192 维质心） | 单向量 | **1s 子段 tokens [N,192]**（N=3-10+） |
| 网络 | GRU 2×128 + Linear | GRU 1×128 → FiLM → GRU 1×128 + Linear | GRU 2×128 + 手写 2 头交叉注意力（帧特征投影作 query，tokens 作 K/V，残差融合）+ FiLM（mask 均值池化 tokens 调制） |
| 参数量 | 253,827 | 260,387 | 390,147 |

v4 与 v3 架构完全相同，差异全在训练数据与选模（见 3.4/3.5）。
v5 接口：`feats [B,T,80] + enroll_tokens [B,N,192] + enroll_mask [B,N]`（True=padding，
C++ 侧 mask 全 False 即无 padding；tokens 为空时回退单 token 质心）。

**v5 的诚实结论**：①收益来源是**多帧条件信息整体**——交叉注意力的注意力权重
**没有可解释的分离模式**（不是"模型学会了挑 enrollment 里哪一段像当前帧"这种故事），
把 tokens 当整体条件后增广误打断 16.4%→10.0%（历代最佳），干净条件与 v4 持平
（93.0% vs 94.0%，−1.0pp 在验收线内）；②v5 流式版（EMA-CMVN 微调）**增广 −17pp
不达标**，实时路径继续用 pvad_v4_stream.onnx，这是"离线 v5 / 实时 v4_stream"
双模型部署的原因（见 `models/pvad/pvad_v5_stream.md` 的如实记录）。

**为什么不是交叉注意力**：192 维单向量作单 token KV 时，softmax(1 token)≡1，
注意力输出退化为 V·emb（线性加性条件），严格弱于 FiLM 的乘性+加性调制。
`PvadModelAttn`（291K）已在 `train_pvad.py --cond attn` 实现但未训练；要让注意力生效
需要 enrollment 的多帧表示（CAM++ 中间层），列入路线图。

### 3.3 损失权重 [1,2,3] 的实验依据

加权 CE（静音/非目标/目标），权重经三轮实验确定：

- 无 target-absent 负样本 + w=[1,1,4]：模型**忽略 enrollment**（随机 embedding 输出不变），
  学成"语音即目标"，FAR≈0.99——负样本是必要条件，不是优化项；
- 有负样本 + w=[1,1,4]：FAR 随训练反而恶化（ep1 0.50→ep2 0.53），过抑制优先
  被类不平衡（目标帧占 ~90%）放大；
- 有负样本 + **w=[1,2,3]**：过抑制仍 3 倍优先，同时给干扰类 2 倍 FAR 压力，收敛最好。

### 3.4 数据（target-absent 与硬负例）

混合物合成：目标+干扰（不同人），SNR ∈ [-5,10]dB，3–8s；10ms 帧级三分类标签由拼接
位置精确生成，重叠帧计为 2。四个版本：v1 干净（含 21% target-absent 负样本）；
v2 加 RIR（70%）+MUSAN（60%）增广；v3 = v1:v2 各 50% + 双干扰硬负例
（两个干扰人同时说话、无目标、标签全 1）+ 50% enrollment 卷 RIR；
**v4 = v3 基础 + 2000 条易混淆硬负例**（干扰人为目标说话人 CAM++ 质心 top5 近邻，
负样本占比 25%：易混淆 2000 + 双干扰 700 + 随机 2116）。
语料 1139 说话人 / 388h（AISHELL-1 + ST-CMDS + Primewords），说话人不重叠划分
train 950 / val 100 / test 89。详见 TRAINING.md 第 2 节。

### 3.5 选模方法论教训：帧级指标 ≠ 门控表现（三次记录）

checkpoint 选择不能只看帧级指标（val loss/F1/FAR），三次实证：

1. **v1**：val_far 跨 epoch 非单调大幅波动（ep1 0.33 → ep2 0.41），帧级 FAR 本身噪声大；
2. **v3**：ep12 的 val FAR 0.317 明显优于按 val F1 保存的 ep4（0.440），但 F1 略低未被选
   ——按 F1 选模错过了 FAR 更优的点（增广 e2e 未达标部分源于此）；
3. **v4**：15 epoch 全存 checkpoint 后，F1−λ·FAR（λ=1/2）与 F1 都指向 ep14，
   但 ep14 的 **val e2e 反而更差**，最终按 val e2e 选 ep11。

结论：**帧级指标与端到端门控表现不总一致**，选模应以 e2e（与门控同参数的端到端评估）
为准；e2e 指标本身也应写入训练目标（路线图）。

---

## 4. 门控状态机

### 4.1 迟滞门控（PvadGate）

帧级 P(target) 过状态机：`p > 0.5` 连续计数，连续 ≥ `confirm`（默认 2）帧触发 INTERRUPT；
`p < 0.5 − hyst`（hyst=0.2）清零复位。与 python 评估（`scripts/eval_pvad.py`）逐参数一致。
5 帧中值滤波在 v3 上收益边际（误打断 7.0%→6.6%），C++ 端未加。

### 4.2 整段 vs 流式（chunked GRU state 复用，当前实现）

PVAD 整段路径（离线/WAV 注入/回归/auto-test，用 `pvad_v4.onnx`）：整段音频一次
fbank + 整段均值归一化 + 单次 GRU 前向，与训练/python 评估完全一致。

**实时流式路径**（CLI `--mic`、qt_demo 麦克风，用 `pvad_v4_stream.onnx`，GRU 隐状态
外置）：每 10ms 帧 O(1) 增量——单帧 fbank（25ms 窗对齐 160 网格）→ EMA CMVN
（α=0.02，会话起始可用 enrollment fbank 均值作先验 `set_cmvn_prior`）→ 每 5 帧
（50ms）一次 chunk 推理，hN 回传为下一 chunk 的 h0。会话起始 0.5s warm-up 不门控
（`--warmup-frames` 可调）；silero VAD speech-end 只复位门控计数、**不重置 GRU/EMA**
（中途重置会造成二次冷启动假阳，实测见 4.4）。

历史包袱（已解决）：旧实时路径是"每 10ms 帧对整段（封顶 8s）重算 fbank+GRU"，
成本随流长增长（实测 1s/2s/4s/8s 段长处单帧 1.56/2.95/5.82/11.54ms，8s 超实时预算，
曾导致 GUI 事件循环饿死）；前缀均值归一化在前 ~0.5s 不稳会把 P(target) 抬高
（非注册 voice2 开头帧 0.49→0.59，TTS 回声 0.50→0.73）。PvadStream 替换后稳态
**0.029 ms/帧（常数级）**，背压机制保留作保护但几乎不再触发。

### 4.3 流式冷启动与 SAPI 域问题（重要，有数值记录）

流式（EMA CMVN + chunk state）集成时发现并修复/记录的问题：

- **冷启动假阳**：EMA 从 0 起步时前 ~0.5-1.5s 特征未归一化，v4s 模型在 **SAPI 合成音**
  上对所有说话人都给高 P(target)（python onnxruntime 双边验证：voice2 前 66 帧、
  echo 前 162 帧 P>0.5）。处置：EMA 用 **enrollment fbank 均值先验**起步
  （`set_cmvn_prior`；python 验证 echo 假阳帧 152→0、voice2 48→9），CLI 用
  `--enroll-wav` 提供，qt_demo 注册时自动累计并随 segments.json 持久化。
- **推理滞后对齐**：chunk=5 推理天然有 0-4 帧滞后，分数必须按**绝对帧号**（`Out.frame`）
  对齐/门控/打时间戳，不能用推入帧号（否则 blip 会越过 warm-up 边界造成误判）。
- **中途重置陷阱**：VAD speech-end 处若重置 GRU/EMA（"每段新会话"），会产生二次冷启动
  假阳（voice1b 在 t≈1.0 的误触发实测来源于此）。正确语义是**整流单会话**（与 python
  eval_stream.py 一致），speech-end 只复位门控 consec。
- **SAPI 域限制（未解）**：即便有先验，v4s 对 SAPI 注册人的目标证据也只在前 ~0.35s
  （voice1b P>0.5 集中在帧 10-35，之后塌缩到 ~0）——0.5s warm-up 会恰好挡住这段证据，
  SAPI 上 A 漏触发。真实人声无此问题（python 流式 e2e 干净 94.5%/增广 82.5%，
  与整段基线仅差 0.5pp）。**SAPI 合成音的流式正确性用整段路径验证**（auto-test 不变），
  流式路径的正确性以真实人声混合物为准。

### 4.4 短窗陷阱（重要，有数值记录）

一个看似自然但**错误**的集成方式：对最近 0.5s 滑窗跑 PVAD、每窗重置 GRU。
实测（python onnxruntime 与 C++ 双边验证，逐帧一致）：非目标说话人 voice2 的
短窗 P(target) 会爬到 **0.9914**（整段形态下同一音频仅 0.003 均值）——单向 GRU 的
输出强烈依赖上下文长度，零状态短窗下模型对非目标给出高置信度假阳性。
**PVAD 必须整段输入、GRU 零初始状态**，这是集成时的第一原则。

---

## 5. C++ 工程架构

### 5.1 模块划分

```
src/  fbank（自实现 80 维 log-mel，radix-2 FFT）  vad（Silero v5，asnorm 用）
      speaker（CAM++ embedding + 模板 v3 存取 + t-norm）  pvad（PVAD ONNX 封装）
      gate（PvadGate 迟滞 / AS-norm Gate）  aec（SpeexDSP，可选）  wav_io
      main（double_voice：离线 --wav / 实时 --mic）
tools/ enroll（模板 v3：正质心 + 负质心列表 + cohort 向量数组）  score  probe
qt_demo/ demo_core（Qt 无关管线核心）  engine（QThread worker）  mainwindow  tts  autotest
scripts/ PVAD 数据/训练/评估 python 管线（见 TRAINING.md）
```

### 5.2 线程模型（qt_demo）

miniaudio 采集回调线程 → 互斥队列 → Engine 所在 QThread（QTimer 5ms 消费 + PVAD 推理）
→ Qt 信号槽回 GUI；TTS 合成也在 Engine 线程（防卡 UI）；播放回调独立线程。
CLI 为单线程管线（原型取舍）。

### 5.3 ONNX Runtime 集成

- MinGW 直接链接 MSVC 预编译 `onnxruntime.dll`（C API 按名导入，header-only C++ 包装）；
  构建后 dll 拷贝到 exe 旁；
- MSVC/qt_demo 链接 `.lib` 导入库；**sherpa-onnx 与主项目 onnxruntime.dll 同名**，
  qt_demo 统一使用 sherpa 包自带的新版 dll（ORT C API 向后兼容），MinGW CLI 仍用 1.20.1；
- pvad session 日志级别设为 ERROR（屏蔽导出示例维度 137 导致的 shape 校验警告）。

### 5.4 已知 MSVC 差异点

`unique_ptr<IncompleteType>` 成员要求显式声明/定义构造与析构（MSVC 在隐式 ctor
实例化时拒绝不完整类型，MinGW 放行）；Windows 头文件的 `far` 宏与字段名冲突（改名
`far_wav`）；源文件 UTF-8 需 `/utf-8`。

---

## 6. 评估方法论与实测数据

### 6.1 方法论

- **帧级**（`eval_pvad.py --frames`）：目标类 recall/precision/F1、FAR=P(判2|真1)、acc，
  按 SNR 分档；test 池 89 人，与训练/验证零重叠；
- **端到端双讲**（`--e2e`，500 条含重叠样本）：P(target) 过 confirm=2 迟滞门控，
  触发帧落在双讲区间 ±20 帧为正确，无目标处触发为误打断，未触发为漏打断；
  对照组为 CAM++ AS-norm 门控的 python 复现（同门控参数）；
- **C++ 一致性**（`scripts/eval_cpp_pvad.py`）：enrollment 走 C++ enroll、管线走
  double_voice，判定逻辑与 python 相同。

### 6.2 帧级指标（ALL 档）

| 测试集 | 模型 | recall | F1 | FAR | acc |
|---|---|---|---|---|---|
| v1 干净 | v1 | 0.976 | 0.936 | 0.350 | 0.901 |
| | v2 | 0.949 | 0.926 | 0.320 | 0.886 |
| | v3 | 0.965 | **0.941** | 0.276 | **0.910** |
| | **v4** | 0.940 | — | **0.210** | — |
| | **v5** | 0.891 | — | 0.225 | — |
| v2 增广 | v1 | 0.951 | 0.876 | 0.683 | 0.779 |
| | v2 | 0.967 | **0.901** | 0.555 | 0.836 |
| | v3 | 0.958 | 0.900 | 0.526 | **0.834** |
| | **v4** | 0.926 | — | **0.446** | — |
| | **v5** | 0.878 | — | 0.435 | — |

### 6.3 端到端双讲（漏打断 / 误打断 / 正确率，python 口径 500 条）

| 测试集 | v1 | v2 | v3 | v4 | **v5（离线默认）** | AS-norm 对照 |
|---|---|---|---|---|---|---|
| v1 干净 | 0.6% / 8.8% / 90.6% | 0.4% / 14.6% / 85.0% | 0.6% / 7.0% / 92.4% | 1.0% / 5.0% / 94.0% | **1.4% / 5.6% / 93.0%** | 64.8% / 0% / 35.2% |
| v2 增广 | 0.8% / 26.8% / 72.4% | 0.2% / 18.0% / 81.8% | 0.2% / 21.6% / 78.2% | 0.6% / 16.4% / 83.0% | **0.6% / 10.0% / 89.4%** | 74.0% / 0% / 26.0% |

C++ 生产管线口径（`double_voice --batch-list`，RNNoise 默认开，200 条/条件，
详见 [REGRESSION.md](REGRESSION.md)）：v4 干净 1.0% / 1.5% / **97.5%**、
增广 1.5% / 16.0% / **82.5%**；**v5 干净 2.5% / 2.5% / 95.0%、
增广 0.5% / 15.0% / 84.5%**（对照 python 93.0%/89.4%：干净 +2.0pp 线内、增广 −4.9pp
略出 ±3pp 边缘——n=200 时两侧 95% CI 有重叠，且相对排序（v5 增广 > v4 增广）一致）。

### 6.4 C++ 管线验证

- **双讲 50 条（v1 模型存档）**：C++ 88.0% vs python 90.6%（差 2.6pp，容差 5pp 内），
  差异来自 enrollment 现场注册路径与数值路径；
- **200 条回归（v4 + RNNoise 默认开，当前默认配置）**：干净 97.5%（误打断 1.5%）、
  增广 82.5%（误打断 16.0%），完整方法与对照见 [REGRESSION.md](REGRESSION.md)；
- **SAPI 回归**（无 AEC）：voice1b(A) 触发（t=0.10–0.11s，与 python 触发帧一致）、
  voice2(B) / TTS 回声不触发（v3：峰值 0.484 / 0.496）；
- **qt_demo --auto-test 三场景**（v4 + RNNoise 默认开）：TTS 自回声不打断 /
  voice1b 打断并停止播放 / voice2 不打断，ALL PASS；
- **AS-norm 噪声三档**（第二代方案存档数据）：A 的 z P95 干净/SNR20/SNR10 =
  21.0/21.8/19.7（稳定），纯余弦 sA P95 = 0.702/0.635/0.529（SNR10 跌破阈值）。

---

## 7. 实时降噪（RNNoise）

### 7.1 选型与集成

- **选型 RNNoise 而非 GTCRN**：RNNoise 单文件模型内置（~1MB vendor 源码、零外部依赖、
  无 ONNX 运行时开销、GRU 极小），GTCRN 指标更强但需额外模型文件 + ORT 会话，
  而本系统瓶颈不在降噪强度（见 7.2 数据——判定几乎不受降噪影响）。RNNoise 是当前
  性价比最高的前置件；若未来确需更强降噪，GTCRN/DPCRN 可作为 `Denoise` 类的替换实现。
- **48k↔16k 重采样**：RNNoise 原生 48kHz/480 采样帧，用 speexdsp 的 `speex_resampler`
  （quality=5）双向转换。16k→48k 恰为 1:3 整数比，每 160 采样输入稳态产 480，
  内部两个小队列对齐帧边界；新增稳态延迟 ≈ **11.4ms**（RNNoise 10ms 帧 + 双向
  重采样群延迟 ~1.4ms）。
- **vendor 方式**：`third_party/rnnoise`（xiph/rnnoise master）源码直接编入，不跑其
  构建系统；master 已将内部 FFT 符号改为 `rnn_fft_*` 前缀，与 speexdsp 的 kiss_fft
  无冲突。MSVC 小修：`_USE_MATH_DEFINES`（M_PI）+ 3 处 C99 VLA 改定长数组
  （pitch.c ×2、celt_lpc.c ×1，均加上限断言）；qt_demo 的 `project()` 需声明
  `LANGUAGES C CXX`（否则 .c 被当作 None 不编译）。
- **管线位置**：采集 → **降噪** → (可选 AEC) → VAD/PVAD 门控。CLI `--denoise rnnoise|off`
  （**默认 rnnoise**）；qt_demo 监听区 `启用降噪` 勾选（**默认勾选**）。离线 --wav 与
  批量模式开降噪时，**每个文件用一个全新 Denoise 实例**先整段预降噪再进管线
  （有状态前件按文件隔离，见 7.3 的教训），保证 PVAD 整段预计算也作用在干净信号上。
- **开销实测**（`--bench-denoise`，Release 单线程）：**0.061 ms/帧**（10ms 帧，
  占实时预算 ~0.6%），延迟 +11.4ms。开销可忽略。

### 7.2 A/B 数据

**SAPI 噪声集**（pvad_v3 门控，P(target) 峰值 / INTERRUPT 次数，off → rnnoise）：

| 样本 | 条件 | off | rnnoise | 变化 |
|---|---|---|---|---|
| voice1b (A) | clean | 0.676 / 1 | 0.802 / 1 | ✅ A 更稳 |
| | SNR20 | 0.637 / 1 | 0.648 / 1 | ≈ |
| | SNR10 | 0.535 / 1 | 0.447 / 0 | ❌ 漏触发 |
| voice2 (B) | clean | 0.484 / 0 | 0.382 / 0 | ✅ 拒识更稳 |
| | SNR20 | 0.653 / 1 | 0.706 / 1 | ❌ 更接近阈值 |
| | SNR10 | 0.693 / 1 | 0.682 / 1 | ≈ |
| echo_only | clean | 0.245 / 0 | 0.430 / 0 | ❌ 抬升（仍 <0.5） |
| | SNR20 | 0.633 / 1 | 0.696 / 1 | ❌ |
| | SNR10 | 0.753 / 1 | 0.722 / 1 | ≈ |

注意：SAPI 语音加白噪后 B/echo **开不开降噪都会误触发**（PVAD 域外问题，非降噪能解），
降噪对 INTERRUPT 判定的唯一改变是把 voice1b_snr10 从触发变成漏触发。

**双讲混合物**（data/mixtures_v2/test 前 30 条含重叠，真实人声+噪声混响增广，
scripts/eval_cpp_pvad.py 判定）：

| 设置 | 漏打断 | 误打断 | 正确率 |
|---|---|---|---|
| denoise off | 0.000 | 0.200 | 0.800 |
| denoise rnnoise | 0.000 | 0.200 | 0.800 |

判定级**零差异**（逐样本一致）。

### 7.3 结论（默认开 + 回滚预案）

RNNoise 在本系统上的收益整体有限但干净条件为正：①干净条件下让 A 更稳（0.68→0.80）、
B 更低（0.48→0.38），qt_demo auto-test 默认开降噪后同样体现（voice1b max_p
0.676→0.761、voice2 0.484→0.382、TTS 回声 0.186→0.173，三场景 ALL PASS）；
②已知的负面边缘场景：加白噪的 SAPI 语音有失真风险（A 在 SNR10 漏触发
0.535→0.447、echo/B 的 P 被抬升）；③对真实双讲混合物（含噪声混响）判定零改善——
**系统瓶颈是 PVAD 在域外/强干扰下的 FAR，不是可加性底噪**。

决策：**默认开**（利大于弊的场景更常见：干净语音上 A/B 同时改善；开销 0.6% 可忽略）。
**回滚预案**：①运行时——CLI `--denoise off`、qt_demo 取消勾选；②代码级——
`git revert 0556210` 即恢复默认关。若后续在真实环境观察到 SNR10 类
白噪场景的漏触发回退，执行回滚即可。

（集成插曲两则：①MSVC VLA 改定长数组时首版数组上界取错（pitch_search 的 max_pitch
实为 588 而非 384），导致 MSVC 下 rnnoise_process_frame 栈越界崩溃；修正上界并复验，
MinGW 侧全部 A/B 数值修复前后逐位一致。②qt_demo auto-test 跨场景共享 Denoise
实例，重采样器/RNNoise 残态污染后续文件特征（v4 下 voice1b 0.47 漏触发、
CLI 同语义 0.64 触发），修为每文件新实例后与 CLI 逐位一致——
**有状态前件必须按文件/流隔离实例**。）

后续若上 GTCRN 级强降噪，应先确认 PVAD 的输入分布偏移问题（降噪后的语音
与训练分布不一致），而不是期望降噪直接改善判定。

---

## 8. 已知限制与路线图

**已知限制（当前）**

1. **增广条件误打断仍高**：v4 增广 e2e 误打断 16.4%（干净已压到 5.0%/C++ 1.5%），
   强噪声/混响下的假阳性是主要剩余问题；
2. **域外余量**：SAPI 合成音上注册余量薄（整段 voice1b 仅 ~7 帧 P>0.5；流式路径
   在 SAPI 上另有冷启动/证据前移问题，见 4.3 节），真实麦克风语音应更好但需实测；
3. **门控无全局冷却**：连续长语音中可多次触发 INTERRUPT（按段复位）。

**路线图（剩余项）**

1. **enrollment 多帧表示 + 真注意力**：取 CAM++ 中间层多帧表示解决单 token KV 退化，
   `--cond attn` 已就绪；
2. **e2e 写进训练目标**：选模三次证明帧级指标≠门控表现（3.5 节），把 e2e 正确率
   直接作为 checkpoint 选择/早停准则；
3. **真实 tts_server gRPC 接入**：qt_demo 目前用本地 VITS 模型直读（sherpa-onnx），
   生产接回 tts_server 的 gRPC（WSL Linux 服务，Windows 侧接 gRPC 成本是当时绕开的原因）。

**已完成（存档）**：**流式 GRU state 复用**（PvadStream：chunk=5 + EMA CMVN +
enrollment 先验，稳态 0.029 ms/帧 替代旧路径 8s 段长 11.54 ms/帧的超实时重算，
python 流式 e2e 与整段仅差 0.5pp）；F1−λ·FAR 联合选模（v4 已实施并引出 val e2e 选模的最终准则）；
硬负例扩充（v4 易混淆负样本 2000 条，FAR 干净 0.276→0.210）；RNNoise 前置降噪
（默认开）；cohort 归一化（AS-norm 基线，现为对照路径）。
