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

全部 16kHz 单声道，中文为主。系统形态：麦克风 → (可选 AEC) → VAD/帧级检测 →
声纹条件打分 → 迟滞门控 → 停止 TTS 播放。

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

| | v1 / v2 | v3（默认） |
|---|---|---|
| 条件机制 | emb broadcast 拼接到每帧（272 维输入） | **FiLM**：emb → 每层 GRU 输入的 (γ,β) 逐维调制 `h⊙(1+γ)+β` |
| 网络 | GRU 2×128 + Linear | GRU 1×128 → FiLM → GRU 1×128 + Linear |
| 参数量 | 253,827 | 260,387 |

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
位置精确生成，重叠帧计为 2。三个版本：v1 干净（含 21% target-absent 负样本）；
v2 加 RIR（70%）+MUSAN（60%）增广；v3 = v1:v2 各 50% + 双干扰硬负例
（两个干扰人同时说话、无目标、标签全 1）+ 50% enrollment 卷 RIR。
语料 1139 说话人 / 388h（AISHELL-1 + ST-CMDS + Primewords），说话人不重叠划分
train 950 / val 100 / test 89。详见 TRAINING.md 第 2 节。

---

## 4. 门控状态机

### 4.1 迟滞门控（PvadGate）

帧级 P(target) 过状态机：`p > 0.5` 连续计数，连续 ≥ `confirm`（默认 2）帧触发 INTERRUPT；
`p < 0.5 − hyst`（hyst=0.2）清零复位。与 python 评估（`scripts/eval_pvad.py`）逐参数一致。
5 帧中值滤波在 v3 上收益边际（误打断 7.0%→6.6%），C++ 端未加。

### 4.2 整段 vs 流式前缀归一化

PVAD 训练用**整段** per-utterance 均值归一化，GRU 零初始状态跑完整段。两种部署形态：

- **离线/文件注入（精确形态）**：整段音频一次 fbank + 整段均值归一化 + 单次 GRU 前向，
  与训练/python 评估完全一致；
- **实时流式（近似形态）**：流起始至今（封顶 8s）整段重算 GRU，均值归一化只能用
  **前缀统计**。前缀统计在前 ~0.5s 不稳定，会把 P(target) 抬高（实测：非注册 voice2
  整段 0.486 → 流式开头帧 0.593；TTS 回声 0.500 → 0.732）。处置：流式 0.5s warm-up
  （报数但不更新门控）。

### 4.3 短窗陷阱（重要，有数值记录）

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
| | **v3** | 0.965 | **0.941** | **0.276** | **0.910** |
| v2 增广 | v1 | 0.951 | 0.876 | 0.683 | 0.779 |
| | v2 | 0.967 | **0.901** | 0.555 | 0.836 |
| | **v3** | 0.958 | 0.900 | **0.526** | **0.834** |

### 6.3 端到端双讲（漏打断 / 误打断 / 正确率）

| 测试集 | v1 | v2 | v3 | AS-norm 对照 |
|---|---|---|---|---|
| v1 干净 | 0.6% / 8.8% / 90.6% | 0.4% / 14.6% / 85.0% | **0.6% / 7.0% / 92.4%** | 64.8% / 0% / 35.2% |
| v2 增广 | 0.8% / 26.8% / 72.4% | 0.2% / 18.0% / 81.8% | 0.2% / 21.6% / 78.2% | 74.0% / 0% / 26.0% |

### 6.4 C++ 管线验证

- **双讲 50 条**：C++ 88.0% vs python 90.6%（差 2.6pp，容差 5pp 内），差异来自
  enrollment 现场注册路径与数值路径；
- **SAPI 回归**（无 AEC）：voice1b(A) 触发（t=0.10–0.11s，与 python 触发帧一致）、
  voice2(B) / TTS 回声不触发（v3：峰值 0.484 / 0.496）；
- **qt_demo --auto-test 三场景**：TTS 自回声不打断 / voice1b 打断并停止播放 /
  voice2 不打断，ALL PASS；
- **AS-norm 噪声三档**（第二代方案存档数据）：A 的 z P95 干净/SNR20/SNR10 =
  21.0/21.8/19.7（稳定），纯余弦 sA P95 = 0.702/0.635/0.529（SNR10 跌破阈值）。

---

## 7. 实时降噪（RNNoise）

### 8.1 选型与集成

- **选型 RNNoise 而非 GTCRN**：RNNoise 单文件模型内置（~1MB vendor 源码、零外部依赖、
  无 ONNX 运行时开销、GRU 极小），GTCRN 指标更强但需额外模型文件 + ORT 会话，
  而本系统瓶颈不在降噪强度（见 8.3 数据——判定几乎不受降噪影响）。RNNoise 是当前
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
  （默认 off）；qt_demo 监听区 `启用降噪` 勾选（默认关）。离线 --wav 开降噪时先整段
  预降噪再进管线，保证 PVAD 整段预计算也作用在干净信号上。
- **开销实测**（`--bench-denoise`，Release 单线程）：**0.061 ms/帧**（10ms 帧，
  占实时预算 ~0.6%），延迟 +11.4ms。开销可忽略。

### 8.2 A/B 数据

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

### 8.3 结论（默认开 + 回滚预案）

RNNoise 在本系统上的收益整体有限但干净条件为正：①干净条件下让 A 更稳（0.68→0.80）、
B 更低（0.48→0.38），qt_demo auto-test 默认开降噪后同样体现（voice1b max_p
0.676→0.761、voice2 0.484→0.382、TTS 回声 0.186→0.173，三场景 ALL PASS）；
②已知的负面边缘场景：加白噪的 SAPI 语音有失真风险（A 在 SNR10 漏触发
0.535→0.447、echo/B 的 P 被抬升）；③对真实双讲混合物（含噪声混响）判定零改善——
**系统瓶颈是 PVAD 在域外/强干扰下的 FAR，不是可加性底噪**。

决策：**默认开**（利大于弊的场景更常见：干净语音上 A/B 同时改善；开销 0.6% 可忽略）。
**回滚预案**：①运行时——CLI `--denoise off`、qt_demo 取消勾选；②代码级——
`git revert` 默认开那次 commit 即恢复默认关。若后续在真实环境观察到 SNR10 类
白噪场景的漏触发回退，执行回滚即可。

（集成插曲：MSVC VLA 改定长数组时首版数组上界取错（pitch_search 的 max_pitch 实为
588 而非 384），导致 MSVC 下 rnnoise_process_frame 栈越界崩溃；修正上界并复验，
MinGW 侧全部 A/B 数值修复前后逐位一致，结论不受影响。）

后续若上 GTCRN 级强降噪，应先确认 PVAD 的输入分布偏移问题（降噪后的语音
与训练分布不一致），而不是期望降噪直接改善判定。

---

## 8. 已知限制与路线图

1. **FAR 偏高**：v3 帧 FAR 0.276/0.526、e2e 误打断 7.0%/21.6%。路线：
   - checkpoint 选择从 val F1 改 **F1−λ·FAR 联合准则**（v3 ep12 val FAR 0.317 优于
     已保存的 ep4 0.440，增广 e2e 未达标部分源于此，零训练成本可改善）；
   - 硬负例扩充（双干扰 2000+ 条、按 enrollment embedding 近邻挑"音色接近"配对）；
   - 门控侧：confirm 调大、触发后全局冷却；
2. **enrollment 单向量瓶颈**：交叉注意力在单 token KV 下退化，取 CAM++ 中间层
   多帧表示后 `--cond attn` 才有意义；
3. **流式 GRU**：当前实时路径每 10ms 整段重算 + 前缀归一化近似；应导出带 state
   输入输出的流式模型（隐藏状态跨步复用）+ 运行 CMVN；
4. **真实 TTS 接入**：qt_demo 目前用本地 VITS 模型直读（sherpa-onnx），生产应接
   tts_server 的 gRPC（WSL Linux 服务，Windows 侧接 gRPC 成本是当时绕开的原因）；
5. **域外余量**：SAPI 合成音上注册余量薄（voice1b 仅 ~7 帧 P>0.5），真实麦克风
   语音应更好但需实测；语料可补 AISHELL-3 / MagicData。
