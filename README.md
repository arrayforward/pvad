# pvad-barge-in：说话人门控打断系统

基于 **Personal VAD** 的说话人门控打断（barge-in）系统：语音交互"双讲"场景下，AI 播放
TTS 时**只有已注册用户 A 的语音可以打断播放**——TTS 自身回声、其他任何人说话都不能打断。
全链路 C++ 实现（16kHz 单声道），含 CLI 工具链与 Qt6 图形 demo。

## 特性

- **PVAD 门控（默认 `--gate pvad`）**：自训练 Personal VAD（GRU + FiLM 条件，260K 参数，
  ONNX），以注册 embedding 为条件做 10ms 帧级三分类（静音/非目标/目标），
  双讲场景帧级识别；AS-norm 门控（CAM++ + cohort t-norm + 负模板 margin）完整保留作基线
- **无 AEC 设计**：不依赖回声参考信号（规避播放/采集时钟对齐难题），抗回声由
  PVAD/负模板门控内生解决（SpeexDSP AEC 保留为可选 `--aec`）
- **实时降噪（RNNoise，默认开）**：vendor xiph/rnnoise 源码编入，16k↔48k 双向重采样
  （speex_resampler），CLI `--denoise rnnoise|off`、qt_demo `[启用降噪]` 默认勾选。
  **回滚方式**：①运行时 `--denoise off` / 取消勾选；②代码级 `git revert 0556210`。
  已知边缘场景：SAPI 合成音+SNR10 白噪下 A 可能漏触发（A/B 数据见
  [docs/DESIGN.md](docs/DESIGN.md) 第 7 节）
- **五种模型版本**：pvad.onnx (v1) / pvad_v2.onnx (增广) / pvad_v3.onnx (FiLM) /
  pvad_v4.onnx (FiLM) / pvad_v5.onnx (多帧 tokens + 交叉注意力，**离线默认**——
  增广历代最佳；实时流式仍用 pvad_v4_stream.onnx，v5 流式不达标)，接口可检测自动适配
- **工具链**：enroll（注册/批量注册）、score（打分）、double_voice（离线/实时/批量回归管线）、
  pvad_demo（Qt6 图形 demo：引导式三段注册向导、TTS 朗读、麦克风/WAV 注入监听、
  `--auto-test`/`--wizard-test`/`--record-test`/`--probe-audio` 无头验证）

## 系统效果（test 池 89 人，与训练/验证零重叠；漏打断 / 误打断 / **正确率**）

python 评估口径（500 条含双讲样本/条件）：

| 测试集 | **PVAD v5（离线默认）** | PVAD v4 | PVAD v3 | AS-norm 基线 |
|---|---|---|---|---|
| v1 干净 | 1.4% / 5.6% / **93.0%** | 1.0% / 5.0% / **94.0%** | 0.6% / 7.0% / 92.4% | 64.8% / 0% / **35.2%** |
| v2 增广（噪声+混响） | 0.6% / 10.0% / **89.4%** | 0.6% / 16.4% / 83.0% | 0.2% / 21.6% / 78.2% | 74.0% / 0% / 26.0% |

v5 = 多帧 enrollment tokens + 交叉注意力：干净与 v4 基本持平（−1.0pp，验收线内），
增广历代最佳（+6.4pp，误打断 16.4%→10.0%）。实时路径仍用 pvad_v4_stream.onnx
（v5 流式版增广不达标，见 [docs/DESIGN.md](docs/DESIGN.md) 第 3 节）。

C++ 生产管线口径（`double_voice --batch-list`，RNNoise 默认开，200 条/条件，
见 [docs/REGRESSION.md](docs/REGRESSION.md)）：

| 模型 | 测试集 | 漏打断 | 误打断 | 正确率 |
|---|---|---|---|---|
| v4 | v1 干净 | 1.0% | 1.5% | **97.5%** |
| v4 | v2 增广 | 1.5% | 16.0% | **82.5%** |
| v5 | v1 干净 | 2.5% | 2.5% | **95.0%** |
| v5 | v2 增广 | 0.5% | 15.0% | **84.5%** |

v4 帧级（干净/增广）：recall 0.940/0.926，FAR 0.210/0.446。
完整数据与方法论见 [docs/DESIGN.md](docs/DESIGN.md) 第 6 节。

## 架构

```
                ┌───────────────────────── 注册（一次性） ─────────────────────────┐
 注册 WAV/录音 ─►│ CAM++ (campplus.onnx) ─► 192 维 enrollment embedding (L2 归一化)│
                └──────────────────────────────┬─────────────────────────────────┘
                                                 │ emb [B,192]
 麦克风 ─► RNNoise 降噪(默认开) ─► fbank 80 维 ─► 均值归一化 ─► feats [B,T,80]
                                                 │
                                                 ▼
                              PVAD (pvad_v5.onnx, GRU+FiLM+交叉注意力)
                                                 │ logits [B,T,3] → softmax → P(target)
                                                 ▼
                          门控（连续 2 帧 >0.5，迟滞清零）──► INTERRUPT ─► 停止 TTS 播放
                                                 ▲
 TTS 文本 ─► sherpa-onnx VITS ─► 16k PCM ─► 播放（miniaudio）
```

同一管线核心（`src/`）供两种前端使用：**CLI**（`double_voice`，离线 --wav / 实时 --mic /
批量回归 --batch-list）与 **qt_demo**（Qt6 Widgets，`pvad_demo`，共用 fbank/speaker/pvad/
denoise/gate 源码，MSVC 构建）。

## 目录结构

```
CMakeLists.txt          # 主项目（MinGW + Ninja）
src/                    # 管线: fbank/vad/speaker/pvad/denoise/aec/gate/wav_io + double_voice
tools/                  # enroll (含 --batch-jsonl 批量注册) / score / probe / gen_cohort.ps1
qt_demo/                # Qt6 Widgets demo（MSVC 构建，含无头验证模式）
scripts/                # PVAD 训练/评估/数据管线 + run_regression_cell.py（C++ 回归驱动）
models/                 # 全部模型权重：campplus.onnx (28MB)、silero_vad.onnx、pvad/ (v1-v4)
test_audio/             # SAPI 合成测试音频 + 注册模板（8MB，可复现验证）
third_party/            # miniaudio.h、speexdsp/、rnnoise/（均为源码编入）
docs/                   # USAGE.md（使用）、DESIGN.md（设计）、REGRESSION.md（回归数据）
TRAINING.md             # PVAD 训练全流程文档（环境/数据/训练/评估/复现）
requirements.txt        # python 训练环境依赖
LICENSE                 # MIT
```

## 快速开始

### 1. 第三方依赖（二进制，不进仓库）

```bash
cd <项目根>/third_party
# ONNX Runtime 1.20.1 (MSVC 预编译，MinGW 也可直接链接其 dll)
curl -L -o /tmp/ort.zip https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-win-x64-1.20.1.zip
unzip /tmp/ort.zip && mv onnxruntime-win-x64-1.20.1 onnxruntime
# sherpa-onnx v1.13.6 (TTS，仅 qt_demo 需要)
curl -L -o /tmp/sherpa.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.6/sherpa-onnx-v1.13.6-win-x64-shared-MD-Release.tar.bz2
tar xjf /tmp/sherpa.tar.bz2 && mv sherpa-onnx-v1.13.6-win-x64-shared-MD-Release sherpa-onnx
```

模型权重（campplus/silero_vad/pvad v1-v4 全部 onnx）已随仓库提交，无需下载。

### 2. CLI（MinGW + Ninja）

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
# 注册 + 打分 + 离线验证（test_audio 自带样本即可跑通，默认 pvad_v4 + RNNoise）
./build/enroll.exe --out test_audio/tpl.bin --pos test_audio/voice1.wav test_audio/voice1b.wav --neg test_audio/tts_ref.wav
./build/score.exe test_audio/tpl.bin test_audio/voice2.wav
./build/double_voice.exe --wav test_audio/voice1b.wav --template test_audio/tpl.bin            # 预期触发 INTERRUPT
./build/double_voice.exe --wav test_audio/voice2.wav --template test_audio/tpl.bin             # 预期不触发
```

### 3. Qt demo（MSVC，需 Qt 6.8.3 msvc2022_64）

```bash
cmake -S qt_demo -B qt_demo/build -G "Visual Studio 17 2022" -DCMAKE_PREFIX_PATH=D:/tools/Qt/6.8.3/msvc2022_64
cmake --build qt_demo/build --config Release
./qt_demo/build/Release/pvad_demo.exe --auto-test     # 无头三场景验证（ALL PASS）
```

TTS 使用本地 VITS 模型（默认路径 `D:/vit/tts/models/vits-piper-zh_CN-huayan-medium`，
可用 `--tts-model` 或 CMake 变量 `TTS_MODEL_DIR` 覆盖），不走 gRPC server。

## 文档导航

- [docs/USAGE.md](docs/USAGE.md) — 使用文档：环境准备、CLI 全参数详解、典型工作流、qt_demo 指南、常见问题
- [docs/DESIGN.md](docs/DESIGN.md) — 设计文档：问题定义、设计决策史、PVAD 模型设计、门控状态机、工程架构、全部实测数据
- [docs/REGRESSION.md](docs/REGRESSION.md) — C++ 生产管线端到端回归数据与测试方法
- [TRAINING.md](TRAINING.md) — 训练文档：语料获取（1139 人/388h）、数据管线 v1→v4、训练配置、评估方法论、复现清单
- `models/pvad/pvad*.md` — 各模型版本的接口约定与指标

## License

MIT，见 [LICENSE](LICENSE)。
