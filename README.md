# pvad-barge-in：说话人门控打断系统

基于 **Personal VAD** 的说话人门控打断（barge-in）系统：语音交互"双讲"场景下，AI 播放
TTS 时**只有已注册用户 A 的语音可以打断播放**——TTS 自身回声、其他任何人说话都不能打断。
全链路 C++ 实现（16kHz 单声道），含 CLI 工具链与 Qt6 图形 demo。

## 特性

- **PVAD 门控（默认）**：自训练 Personal VAD（GRU + FiLM 条件，260K 参数，ONNX），
  以注册 embedding 为条件做 10ms 帧级三分类（静音/非目标/目标），双讲场景帧级识别
- **无 AEC 设计**：不依赖回声参考信号（规避播放/采集时钟对齐难题），抗回声由
  PVAD/负模板门控内生解决（SpeexDSP AEC 保留为可选 `--aec`）
- **可选实时降噪（RNNoise）**：vendor xiph/rnnoise 源码编入，16k↔48k 双向重采样
  （speex_resampler），`--denoise rnnoise` / qt_demo 勾选启用，默认关
  （A/B 实测结论见 [docs/DESIGN.md](docs/DESIGN.md) 第 7 节——对白噪收益有限且
  个别场景退化，仅建议持续底噪的干净会议室场景手动开）
- **AS-norm 门控基线**：CAM++ embedding + cohort t-norm 归一化 + 负模板 margin 判决，
  完整保留用于对照实验（`--gate asnorm`）
- **三种模型版本**：pvad.onnx (v1) / pvad_v2.onnx (增广) / pvad_v3.onnx (FiLM，默认)，
  接口相同可直接替换
- **工具链**：enroll（注册）、score（打分）、double_voice（离线/实时管线）、
  pvad_demo（Qt6 图形 demo，含 `--auto-test` 无头验证）

## 系统效果（test 池 89 人，与训练/验证零重叠）

端到端双讲打断判定（漏打断 / 误打断 / **正确率**，500 条含双讲样本）：

| 测试集 | PVAD v3（默认） | PVAD v1 | PVAD v2 | AS-norm 基线 |
|---|---|---|---|---|
| v1 干净 | 0.6% / 7.0% / **92.4%** | 0.6% / 8.8% / 90.6% | 0.4% / 14.6% / 85.0% | 64.8% / 0% / **35.2%** |
| v2 增广（噪声+混响） | 0.2% / 21.6% / **78.2%** | 0.8% / 26.8% / 72.4% | 0.2% / 18.0% / 81.8% | 74.0% / 0% / 26.0% |

帧级（v3，干净/增广）：recall 0.965/0.958，F1 0.941/0.900，FAR 0.276/0.526。
C++ 管线与 python 评估逐帧等价（50 条双讲样本 C++ 正确率 88.0% vs python 90.6%，
差异在 5pp 容差内）。完整数据见 [docs/DESIGN.md](docs/DESIGN.md) 第 6 节。

## 架构

```
                ┌───────────────────────── 注册（一次性） ─────────────────────────┐
 注册 WAV ─────►│ CAM++ (campplus.onnx) ─► 192 维 enrollment embedding (L2 归一化)│
                └──────────────────────────────┬─────────────────────────────────┘
                                                 │ emb [B,192]
 麦克风 ─► 10ms 帧流 ─► fbank 80 维 ─► 均值归一化 ─► feats [B,T,80]
                                                 │
                                                 ▼
                              PVAD (pvad_v3.onnx, GRU+FiLM)
                                                 │ logits [B,T,3] → softmax → P(target)
                                                 ▼
                          门控（连续 2 帧 >0.5，迟滞清零）──► INTERRUPT ─► 停止 TTS 播放
                                                 ▲
 TTS 文本 ─► sherpa-onnx VITS ─► 16k PCM ─► 播放（miniaudio）
```

## 目录结构

```
CMakeLists.txt          # 主项目（MinGW + Ninja）
src/                    # 管线: fbank/vad/speaker/pvad/aec/gate/wav_io + double_voice 主程序
tools/                  # enroll / score / probe / gen_cohort.ps1
qt_demo/                # Qt6 Widgets demo（MSVC 构建，含 --auto-test 无头验证）
scripts/                # PVAD 训练/评估/数据管线 python 脚本
models/                 # 全部模型权重：campplus.onnx (28MB)、silero_vad.onnx、pvad/ (v1/v2/v3)
test_audio/             # SAPI 合成测试音频 + 注册模板（8MB，可复现验证）
third_party/            # miniaudio.h、speexdsp/（源码编入）
docs/                   # USAGE.md（使用文档）、DESIGN.md（设计文档）
TRAINING.md             # PVAD 训练全流程文档（环境/数据/训练/评估/复现）
requirements.txt        # python 训练环境依赖
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

模型权重（campplus/silero_vad/pvad 全部 onnx）已随仓库提交，无需下载。

### 2. CLI（MinGW + Ninja）

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
# 注册 + 打分 + 离线验证（test_audio 自带样本即可跑通）
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
- [TRAINING.md](TRAINING.md) — 训练文档：语料获取、数据管线、训练配置、评估方法论、复现清单
- `models/pvad/pvad*.md` — 各模型版本的接口约定与指标

## License

（待定 / Placeholder）
