# 架构文档（ARCHITECTURE）

pvad-barge-in 系统的工程架构：双路径（离线/实时）推理管线、模块契约、线程模型、
部署矩阵与构建体系。设计决策与理由见 [DESIGN.md](DESIGN.md)；操作手册见
[USAGE.md](USAGE.md)；训练方案见 [TRAINING.md](../TRAINING.md)；回归数据见
[REGRESSION.md](REGRESSION.md)。

---

## 1. 系统总览

```
═══ 路径 A：离线 / 批量（double_voice --wav / --batch-list、qt_demo WAV 注入/auto-test）═══

 wav/采集 ─► RNNoise 降噪(默认开, 可选 off) ─► fbank 80维 + 整段 per-bin 均值归一化
     │
     ▼
 pvad_v5.onnx（多帧 enrollment tokens + 交叉注意力 + FiLM）
     │   feats [B,T,80] + enroll_tokens [B,N,192] + enroll_mask [B,N] → logits [B,T,3]
     ▼
 PvadGate（P(target) 连续 2 帧 >0.5 触发，<0.3 清零）──► INTERRUPT 事件（停止 TTS 播放）

═══ 路径 B：实时（double_voice --mic、qt_demo 麦克风）═══

 mic ─► RNNoise 降噪(默认开) ─► 单帧 fbank（25ms 窗对齐 160 网格）→ EMA CMVN（α=0.02，
     │  enrollment fbank 均值先验）
     ▼
 pvad_v4_stream.onnx（GRU 隐状态外置，chunk=5 帧/50ms 推理一次，hN 回传 h0；
     │   会话起始 0.5s warm-up 不门控；每帧 O(1) 增量，稳态 0.029 ms/帧）
     ▼
 PvadGate（同参数）──► INTERRUPT 事件
```

两条路径共用：fbank 参数（25ms/10ms/80 mel/preemph 0.97/fft512）、CAM++ enrollment
（campplus.onnx 192 维）、PvadGate 迟滞门控、RNNoise 降噪、miniaudio 采集/播放。
**离线永远用整段路径**（与训练/python 评估完全一致）；**实时用流式路径**（精度代价
python 口径仅 −0.5pp）。

---

## 2. 模块清单与职责

### 2.1 `src/`（管线核心，CLI 与 qt_demo 共用源码）

| 模块 | 职责 | 关键接口 |
|---|---|---|
| `fbank.h/.cpp` | 自实现 80 维 log-mel（预加重 0.97 → hamming → radix-2 FFT → mel → log），与 python `scripts/pvad_common.py` 逐参数对齐 | `Fbank::compute(pcm, n, out) -> int T`；`compute_one(win400, out80)`（单帧） |
| `speaker.h/.cpp` | CAM++ embedding + 模板 v4 存取 + t-norm 统计 | `embed(pcm, n) -> [192]`；`embed_tokens(pcm, n) -> [N][192]`（1s 切分）；`save/load_template` |
| `vad.h/.cpp` | Silero VAD v5（ONNX，64 采样 context 拼接用法） | `Vad::process(frame160, n) -> float prob` |
| `pvad.h/.cpp` | PVAD 整段推理，**双接口自动检测**（v1-v4 单向量 / v5 enroll_tokens+mask） | `target_probs(feats, T, emb, tokens*) -> [P(target)×T]` |
| `pvad_stream.h/.cpp` | 流式 PVAD：GRU state 复用 + EMA CMVN + 先验 | `set_emb/set_cmvn_prior/set_warmup`；`push_frame(fbank80) -> Out{p,valid,gated,frame}`；`reset()` |
| `gate.h` | 门控状态机 | `PvadGate::update(p) -> bool fire`（confirm/hyst）；`Gate`（AS-norm z+margin，对照路径） |
| `denoise.h/.cpp` | RNNoise 实时降噪（speex_resampler 16k↔48k，帧界对齐队列） | `Denoise::process(in160, out160)`；0.061 ms/帧，+11.4ms 延迟 |
| `aec.h/.cpp` | SpeexDSP MDF 回声消除（**可选 `--aec`，默认关**） | `Aec::process(near, far, out)` |
| `wav_io.h` | 极简 WAV 读写（16k 16bit PCM / 32bit float 单声道） | `read_wav(path) -> WavData`；`write_wav16` |

### 2.2 `tools/`（CLI 工具）

- `enroll`：注册（正质心 + 负质心 + cohort + v4 tokens）；`--batch-jsonl` 批量注册（回归 harness 用）
- `score`：整段声纹打分（sA_raw / sA_norm z / sNeg + 判决，AS-norm 口径）
- `probe`：打印任意 ONNX 模型的输入输出名/shape/dtype

### 2.3 `qt_demo/`（Qt6 Widgets 图形程序，MSVC 构建）

| 层 | 类 | 职责 |
|---|---|---|
| 核心（Qt 无关） | `DemoCore` | 注册状态（逐段 SegRecord：emb/tokens/fbank 先验）+ 整段 PVAD 预计算 + PvadGate |
| | `WizardController` | 引导式三段注册状态机（备份/恢复旧注册，取消语义精确） |
| | `EnrollStore` | 注册持久化：`enrollment/tpl.bin`（CLI v4 模板格式，双向兼容）+ `segments.json`（逐段 embedding/tokens/fbank_mean，%.9g 无损往返） |
| | `UiState`/`Backpressure`（ui_state.h） | 按钮使能三态 + 音频队列背压决策（可无头测试） |
| 线程层 | `Engine`（QObject，独立 QThread） | miniaudio 采集/播放设备、流式 PVAD 推理、TTS 合成、信号槽回 GUI |
| GUI 层 | `MainWindow` | 注册区（WAV/录音/向导/持久化状态）、TTS 区、监听区（麦克风/WAV 注入/降噪勾选）、门控状态、事件日志 |
| 无头验证 | `autotest` | `--auto-test`（三场景打断回归）、`--wizard-test`、`--persist-test`、`--record-test`、`--ui-state-test`、`--probe-audio` |

---

## 3. 关键接口契约

### 3.1 PVAD ONNX 接口代际

| 代际 | 输入 | 输出 | 用途 |
|---|---|---|---|
| v1-v4（单向量） | `feats [B,T,80]` + `emb [B,192]` | `logits [B,T,3]` | 离线（v4 及更早） |
| **v5（多帧注意力）** | `feats [B,T,80]` + `enroll_tokens [B,N,192]` + `enroll_mask [B,N]`（True=padding，C++ 全 False） | `logits [B,T,3]` | **离线默认** |
| stream（state 外置） | `feats_chunk [B,t,80]` + `emb [B,192]` + `h0 [2,B,128]` | `logits [B,t,3]` + `hN [2,B,128]` | **实时默认（v4_stream）** |

共同约定：feats 为 25ms/10ms 80 维 log-mel；离线 per-utterance per-bin 均值归一化，
流式 EMA CMVN α=0.02（调用方负责）；emb/tokens 均 L2 归一化 192 维。

### 3.2 模板格式（tpl.bin）演进

| 版本 | 布局 | 读兼容 |
|---|---|---|
| v1 | `int32 dim + float pos[dim]` | 已废弃（拒绝） |
| v2 | version=2, dim, n_neg, pos, 负质心(label+centroid) | 已废弃（拒绝） |
| v3 | + n_cohort + cohort[N,dim] | load 支持（tokens 为空） |
| **v4（当前）** | + n_tok + tokens[N,dim] | load 支持 v3+v4；save 恒写 v4 |

qt_demo 的 `enrollment/tpl.bin` 与 CLI `enroll` 产物完全同格式，双向互用。

### 3.3 segments.json（qt_demo 注册持久化）

```json
{"fbank_mean": [80 floats],           // 可选：流式 CMVN 先验（enrollment fbank 均值）
 "segments": [
  {"wav": "recordings/rec_20260827_101530.wav",
   "duration_s": 3.2, "time": "2026-08-27 10:15:30",
   "embedding": [192 floats],          // %.9g 无损往返
   "tokens": [[192 floats], ...]}      // v5：1s 子帧 tokens（旧格式可无 -> wav 重算/降级）
 ]}
```

加载三态：tokens 在 → 用；缺 → 从 wav 重算（recordings/ 相对路径解析）；wav 已删 →
告警并回退单 token 质心（v5 模型兼容）。

---

## 4. 线程与实时模型（qt_demo）

- **三线程**：miniaudio 采集回调线程（只入队）→ Engine 所在 QThread（QTimer 5ms 消费：
  降噪 → 单帧 fbank → PvadStream chunk 推理 → 门控 → 信号槽）→ GUI 线程（信号槽更新）。
  TTS 合成也在 Engine 线程（防卡 UI）；播放回调独立线程。
- **背压机制（事件循环饿死事故的教训）**：旧流式路径每 10ms 对整段（封顶 8s）重算
  fbank+GRU，8s 段长单帧 11.54ms 超实时预算，采集队列无限积压、tick() 不返回、
  事件循环饿死（"停止/朗读按钮全部没反应"）。修复：积压 >1s 丢弃最旧帧（保留最新
  0.5s）+ 每 tick 最多处理 5 帧（有界返回）+ 处理率看门狗告警。流式 GRU state 复用
  上线后单帧降到常数级，背压仅作保护保留。
- **流式性能**（`--bench-stream`，Release 单线程）：旧路径 1s/2s/4s/8s 段长 =
  1.56/2.95/5.82/11.54 ms/帧（随流长增长）；**PvadStream = 0.029 ms/帧（常数级，
  约 400×）**。推理有 0-4 帧 chunk 滞后，分数按绝对帧号对齐（Out.frame）。

---

## 5. 部署矩阵（最终版）

| 场景 | 模型 | 说明（python 口径 干净/增广） |
|---|---|---|
| 离线 / 批量回归 | `pvad_v5.onnx` | 多帧 tokens+交叉注意力，93.0% / 89.4%（增广历代最佳） |
| 实时：强噪 / 混响 | `pvad_v4_stream.onnx` | EMA-CMVN 微调，94.5% / 82.5%（增广鲁棒性最好） |
| 实时：干净场景（可选） | `pvad_v5s_stream.onnx` | cos 注意力+EMA 微调，92.0% / 85.0%（增广差 2pp 未过线） |

### 模型文件清单（models/）

| 文件 | 接口 | 状态 | 用途 |
|---|---|---|---|
| `campplus.onnx` (28MB) | fbank→192d emb | 生产 | CAM++ enrollment（3D-Speaker 中文） |
| `silero_vad.onnx` (2.3MB) | audio+state→prob | 生产 | VAD（asnorm 门控与 CLI 流式 speech-end 检测） |
| `pvad/pvad.onnx` | 单向量 | 存档 | v1 基线 |
| `pvad/pvad_v2.onnx` | 单向量 | 存档 | v2（增广数据） |
| `pvad/pvad_v3.onnx` | 单向量 | 存档 | v3（FiLM） |
| `pvad/pvad_v4.onnx` | 单向量 | 存档 | v4（硬负例；前离线默认） |
| `pvad/pvad_v5.onnx` | tokens+mask | **生产（离线默认）** | 多帧注意力，93.0/89.4 |
| `pvad/pvad_v4_stream.onnx` | state 外置 | **生产（实时默认）** | EMA-CMVN 微调，94.5/82.5 |
| `pvad/pvad_v5s_stream.onnx` | state 外置 | 可选 | cos 注意力，92.0/85.0（干净场景可选） |
| `pvad/pvad_v5_stream.onnx` | state 外置 | **禁用** | v5 流式版：增广 −17pp 不达标 |
| `pvad/pvad_v6.onnx`/`pvad_v6_stream.onnx` | tokens+mask | 终止 | per-frame CMVN：抹掉段级能量动态，增广 77.5%，方向终止 |

`.onnx.data` 为外部权重，必须与同名 `.onnx` 同目录。`best*.pt` 为训练 checkpoint
（fine-tune 入口，见 TRAINING.md）。

---

## 6. 构建体系

双工具链互不干扰：

| | CLI（double_voice/enroll/score/probe） | qt_demo（pvad_demo） |
|---|---|---|
| 工具链 | MinGW-w64 g++ + cmake + Ninja | MSVC 2022（VS/BuildTools）+ Qt 6.8.3 msvc2022_64 |
| 配置 | `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release` | `cmake -S qt_demo -B qt_demo/build -G "Visual Studio 17 2022" -DCMAKE_PREFIX_PATH=<Qt>/6.8.3/msvc2022_64` |
| ORT 链接 | 直接链接 `onnxruntime.dll`（C API 按名导入） | sherpa 包自带新版 `onnxruntime.dll` + `.lib` 导入库 |

第三方依赖（大体积二进制不进 git，下载命令见 USAGE.md；源码编入的在 third_party/）：

| 依赖 | 形式 | 用途 |
|---|---|---|
| ONNX Runtime 1.20.1 | 预编译 zip（github microsoft/onnxruntime） | 全部 ONNX 推理 |
| sherpa-onnx v1.13.6 | 预编译 tar.bz2（github k2-fsa） | qt_demo TTS（本地 VITS 模型直读，不走 gRPC server） |
| miniaudio.h | 单头文件 | 采集/播放 |
| speexdsp 1.2.1 | 源码编入（mdf/resample 等） | 可选 AEC + RNNoise 重采样 |
| rnnoise (xiph master) | 源码编入 | 实时降噪（默认开） |
| Qt 6.8.3 msvc2022_64 | 官方安装 | qt_demo GUI（windeployqt 部署） |
