# 使用文档（USAGE）

本文档覆盖 pvad-barge-in 系统的环境准备、CLI 工具全参数、典型工作流与 qt_demo 使用。
设计原理见 [DESIGN.md](DESIGN.md)，训练见 [TRAINING.md](../TRAINING.md)。

---

## 1. 环境准备

### 1.1 工具链

| 用途 | 工具链 | 说明 |
|---|---|---|
| CLI（double_voice/enroll/score/probe） | MinGW-w64 g++（ucrt）+ cmake + ninja | `cmake -B build -G Ninja` |
| qt_demo | MSVC 2022（VS2022 或 BuildTools）+ Qt 6.8.3 msvc2022_64 | `cmake -G "Visual Studio 17 2022"`，Qt 路径通过 `-DCMAKE_PREFIX_PATH` 指定 |
| PVAD 训练/评估（可选） | Python 3.12 + venv | 见 [TRAINING.md](../TRAINING.md) 第 1 节；`.venv` 被删除时按该节重建 |

两套 C++ 工具链互不影响：CLI 用 MinGW 直接链接 MSVC 预编译的 `onnxruntime.dll`
（C API 按名导入）；qt_demo 用 MSVC 原生链接。

### 1.2 第三方依赖下载（大体积二进制，不进 git）

在 `third_party/` 下执行：

```bash
cd <项目根>/third_party

# ONNX Runtime 1.20.1（CLI 与 qt_demo 都需要；qt_demo 实际加载 sherpa 包自带的新版 dll）
curl -L -o /tmp/ort.zip https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-win-x64-1.20.1.zip
unzip /tmp/ort.zip && mv onnxruntime-win-x64-1.20.1 onnxruntime

# sherpa-onnx v1.13.6 win-x64 shared-MD-Release（仅 qt_demo 的 TTS 需要）
curl -L -o /tmp/sherpa.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.6/sherpa-onnx-v1.13.6-win-x64-shared-MD-Release.tar.bz2
tar xjf /tmp/sherpa.tar.bz2 && mv sherpa-onnx-v1.13.6-win-x64-shared-MD-Release sherpa-onnx
```

### 1.3 模型文件清单（已随仓库提交）

| 文件 | 大小 | 用途 |
|---|---|---|
| `models/campplus.onnx` | 28MB | CAM++ 声纹 embedding（192 维，3D-Speaker 中文训练） |
| `models/silero_vad.onnx` | 2.3MB | Silero VAD v5（asnorm 门控的语音检测） |
| `models/pvad/pvad.onnx(+ .data)` | ~1MB | PVAD v1（emb 拼接条件） |
| `models/pvad/pvad_v2.onnx(+ .data)` | ~1MB | PVAD v2（同 v1 架构，增广数据） |
| `models/pvad/pvad_v3.onnx(+ .data)` | ~1MB | PVAD v3（FiLM 条件） |
| `models/pvad/pvad_v4.onnx(+ .data)` | ~1MB | PVAD v4（FiLM 条件，**默认**：双条件双优，C++ 干净回归 97.5%） |
| `models/pvad/best*.pt` | 各 ~1MB | 训练 checkpoint（fine-tune 入口，TRAINING.md 复现用） |

`.onnx.data` 是外部权重文件，必须与同名 `.onnx` 放在同一目录，不要单独移动/改名。

### 1.4 TTS 模型（仅 qt_demo）

默认读 `D:\vit\tts\models\vits-piper-zh_CN-huayan-medium`（含 `model.onnx`、`tokens.txt`、
`espeak-ng-data`，vits-piper 中文 huayan 音色，22050Hz）。更换：qt_demo 传
`--tts-model <dir>`，或 CMake 配置 `-DTTS_MODEL_DIR=<dir>`。

---

## 2. CLI 工具详解

所有命令默认从**项目根目录**运行（模型相对路径默认 `models/`，均可用参数覆盖）。
WAV 输入统一要求 **16kHz 16bit PCM（或 32bit float）单声道**。

### 2.1 enroll — 注册

```bash
./build/enroll.exe --out tpl.bin [--model models/campplus.onnx] \
    [--pos] a1.wav a2.wav ... [--neg tts1.wav ...] [--cohort c1.wav ...]
```

| 参数 | 含义 | 建议 |
|---|---|---|
| `--out` | 模板输出路径（v3 二进制格式） | `tpl.bin` |
| `--model` | CAM++ 模型路径 | 默认 `models/campplus.onnx` |
| `--pos` | 后续裸 wav 归为正样本（注册用户 A）。可多次使用切换分组 | 3–10s 干净语音 × 2 段以上 |
| `--neg` | 后续 wav 归为负样本，组内合并为一个负质心（label `neg`） | 放 TTS 引擎的语音样本；多音色则分多次 enroll 或各给一个 |
| `--cohort` | 后续 wav 各存一个独立 embedding（AS-norm 归一化用，不合并） | 15–30+ 个**真实**无关说话人；SAPI 变体不可用（见 5.3） |

正样本处理：每段提 embedding → L2 归一化 → 均值 → 再归一化（质心）。
pvad 门控只用正质心；`--neg`/`--cohort` 仅供 asnorm 门控。

### 2.2 score — 整段打分（asnorm 口径）

```bash
./build/score.exe [--model models/campplus.onnx] [--threshold 0.55] [--margin 0.15] \
    [--z-threshold 3.0] [--norm-topk 50] [--no-norm] tpl.bin test.wav
```

输出整段 `sA_raw`、`sA_norm`（z，含 cohort 数/μ/σ）、`sNeg` 及判决（PASS/REJECT）。
用于验证注册区分度与 asnorm 参数标定。

### 2.3 double_voice — 管线主程序

```bash
./build/double_voice.exe --wav in.wav --template tpl.bin [选项]     # 离线
./build/double_voice.exe --mic --template tpl.bin [选项]            # 实时
```

| 参数 | 默认 | 含义与建议 |
|---|---|---|
| `--wav PATH` | — | 离线模式输入 wav |
| `--mic` | — | 实时模式（miniaudio 采集 16k 单声道） |
| `--template PATH` | `tpl.bin` | enroll 产物 |
| `--gate pvad\|asnorm` | `pvad` | 门控模式。pvad 双讲场景远优于 asnorm |
| `--pvad-model PATH` | `models/pvad/pvad_v4.onnx` | PVAD 模型（默认 v4；版本切换见 4.4） |
| `--pvad-threshold` | 0.5 | P(target) 触发阈值。降 FAR 可试 0.6 |
| `--pvad-hyst` | 0.2 | 低于 threshold−hyst 计数清零 |
| `--confirm` | 2 | 连续 N 次满足才触发。降误打断可试 3–5 |
| `--window-ms` | 500 | asnorm 滑窗长度（pvad 模式不用）。500–800ms 最优 |
| `--threshold` | 0.55 | asnorm `--no-norm` 模式的 sA_raw 阈值 |
| `--z-threshold` | 3.0 | asnorm z 分数阈值（实测 A 的 z P95≥19.7、impostor≤−1.2，余量大） |
| `--margin` | 0.15 | sA_raw − sNeg 最小间隔（负模板压制回声，**不要去掉**） |
| `--norm-topk` | 50 | t-norm cohort top-K |
| `--no-norm` | off | asnorm 退回纯余弦（对比实验用） |
| `--vad-threshold` | 0.5 | silero VAD 语音概率阈值（asnorm 模式门控） |
| `--aec` | off | 显式启用 SpeexDSP AEC（离线需同时 `--far`） |
| `--far PATH` | — | AEC 参考 wav（TTS 播放流） |
| `--denoise rnnoise\|off` | `rnnoise` | 实时降噪（RNNoise，位于 采集→降噪→门控 最前端），**默认开**。<br>**回滚**：①运行时 `--denoise off`；②代码级 `git revert` 默认开那次 commit。<br>已知边缘场景：SAPI 合成音+SNR10 白噪下 A 可能漏触发（0.535→0.447）；真实人声双讲混合物开/关判定零差异（详见 DESIGN.md 第 7 节）。离线模式开降噪会先整段预降噪再跑管线，保证 PVAD 整段预计算作用在干净信号上 |
| `--bench-denoise` | — | 实测降噪单帧耗时后退出（不开管线） |
| `--play-tone` | off | 实时模式播放 440Hz 测试音（模拟 TTS 播放） |
| `--seconds` | 30 | 实时模式运行时长 |
| `--vad-model/--spk-model` | models/ 下 | silero/CAM++ 模型路径 |

**pvad 离线行为**：整段音频一次性算 P(target) 序列（整段均值归一化 + 单次 GRU 前向，
与训练/python 评估一致），再逐帧过门控；PVAD 自带静音类，不经 silero 门控。
**pvad 实时行为**：流起始至今（封顶 8s）整段 GRU + 前缀均值归一化（近似，见 5.2）。

### 2.4 probe — 打印 ONNX 模型输入输出

```bash
./build/probe.exe models/pvad/pvad_v4.onnx
```

---

## 3. 典型工作流

```bash
# 1) 注册（A 的两段语音 + TTS 音色作负样本）
./build/enroll.exe --out tpl.bin --pos a1.wav a2.wav --neg tts_sample.wav

# 2) 注册质量检查（asnorm 口径：sA 应 >0.9，无关人 <0.3）
./build/score.exe tpl.bin a1.wav
./build/score.exe tpl.bin other.wav

# 3) 离线验证（双讲样本/真人录音）
./build/double_voice.exe --wav test.wav --template tpl.bin --pvad-model models/pvad/pvad_v4.onnx
# 观察每帧 p_target、>>> INTERRUPT <<< 及时间戳

# 4) 实时运行
./build/double_voice.exe --mic --template tpl.bin --pvad-model models/pvad/pvad_v4.onnx --seconds 60

# 5) 批量评估（需训练数据环境，见 TRAINING.md）
python scripts/eval_cpp_pvad.py --max-n 50     # C++ 管线端到端对照
```

---

## 4. qt_demo 使用指南

### 4.1 构建与启动

```bash
cmake -S qt_demo -B qt_demo/build -G "Visual Studio 17 2022" -DCMAKE_PREFIX_PATH=<Qt>/6.8.3/msvc2022_64
cmake --build qt_demo/build --config Release
./qt_demo/build/Release/pvad_demo.exe                 # GUI
./qt_demo/build/Release/pvad_demo.exe --auto-test     # 无头三场景验证
./qt_demo/build/Release/pvad_demo.exe --probe-audio   # 采集/播放设备冒烟
```

注意 **`-S qt_demo` 不能省**（否则 CMake 以项目根为源码目录）。Qt DLL 由 post-build 的
windeployqt 自动拷贝；sherpa/onnxruntime DLL 一并拷贝（onnxruntime.dll 用 sherpa 包自带版本）。

### 4.2 界面各区

- **注册区**：
  - `开始引导注册`（**推荐**）：进入 3 段向导——开始时备份旧注册并清空，逐段给出
    场景提示与念词文本（第 1 段正常距离照念 / 第 2 段约 1 米提高音量照念 /
    第 3 段自由发挥 ≥5 秒），界面大号显示「第 N/3 段」+ 提示 + 文本，
    `开始录音`/`停止`（录音中显示秒数，≥3s 才允许手动停止，15s 自动停），
    每段录入后打勾「✓ 已录入（Xs）」并自动进入下一段；3 段完成显示
    「注册完成（3 段）」并退出向导。单段 <2s 拒收、可重录本段（不计入进度）。
    `取消向导` 随时可中途退出并**恢复向导前的旧注册**（取消前的已有注册不受影响）。
    向导中监听与 TTS 朗读按钮置灰（沿用互斥机制）
  - `选择WAV注册A`（可多选 16k WAV）→ 状态显示"已注册 N 段"
  - `录音注册`（手动模式）：点击开始从默认麦克风录音，再次点击停止（15s 上限自动停止），
    过程中显示已录时长；停止后立即过 CAM++ 提 embedding **追加**进注册集合
    （与 WAV 注册同一质心平均逻辑），每段录音同时保存到 `qt_demo/recordings/`
    （16k mono，文件名带时间戳）便于复用排查。<2s 的录音会被拒（提示"建议3-10秒"）
  - `清空注册`：清空注册集合
  - 静态提示：建议每段 3-10 秒、录 3-5 段、变换与麦克风的距离/角度
  - 录音注册与监听互斥（监听中录音按钮置灰，反之亦然），避免采集流冲突
- **TTS 区**：文本框 + `朗读`：合成（Engine 线程，不卡 UI）→ miniaudio 播放；无播放设备时
  自动降级为虚拟播放（状态机一致，便于无头环境）
- **监听区**：`开始监听`/`停止`；音源二选一：
  - **麦克风**：真实采集（流式 PVAD，0.5s warm-up）
  - **WAV 注入**：选 wav 模拟麦克风（整段预计算，可复现；约 2 倍速灌入）
  - `启用降噪` 勾选框（**默认开**）：RNNoise 实时降噪，对麦克风和 WAV 注入都生效，
    新增延迟约 11ms。回滚：取消勾选（运行时），或 `git revert` 默认开那次 commit（代码级）
- **门控状态**：P(target) 进度条 + 数值 + consec；触发时红色高亮
- **事件日志**：注册/合成/播放/监听/INTERRUPT/播放停止，均带时间戳

### 4.3 无头模式

- `--auto-test`：三场景回归（TTS 自回声 / voice1b 打断 / voice2 不打断）。
  场景用 TTS 音频优先读缓存 `test_audio/auto_tts.wav`（VITS 合成带随机性，
  缓存保证判定可复现；删除该文件即重新合成并缓存）
- `--probe-audio`：默认采集/播放设备冒烟
- `--record-test [秒]`：录音注册代码路径无头冒烟（录音 → 存 wav → CAM++ 注册）
- `--wizard-test`：引导注册状态机无头验证（单段太短重录 / 完成 3 段 /
  中途取消恢复旧质心 / 0 段取消边界，均不依赖真实麦克风）

### 4.4 模型版本切换

- CLI：`--pvad-model models/pvad/pvad_v4.onnx`（各版接口相同：`feats[B,T,80]+emb[B,192]→logits[B,T,3]`；
  **默认已是 v4**，此参数仅用于切旧版对比）
- qt_demo：默认编译期写死 `models/pvad/pvad_v4.onnx`（`engine.cpp` 的 `init()`），改路径重编译即可
- **当前默认 v4 的理由**：python 口径干净 94.0%/增广 83.0% 双优（v3 为 92.4%/78.2%），
  C++ 口径干净回归 97.5%（v3 为 95.5%，误打断 4.0%→1.5%）
- 历史版本选择参考 [TRAINING.md](../TRAINING.md) 第 5 节（v2 在强噪声/混响下曾是首选，
  v4 增广 83.0% 已超过 v2 的 81.8%）

---

## 5. 常见问题

### 5.1 `.venv` 消失 / python 脚本报文件不存在

训练环境的 `.venv` 曾被整体删除。重建只需：

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 5.2 实时/流式路径的 warm-up 与 P(target) 虚高

PVAD 训练用**整段均值归一化**；流式只能用前缀归一化，前 ~0.5s 统计不稳会把
P(target) 抬高（实测非注册 voice2 流式开头帧 0.49→0.59，TTS 回声 0.50→0.73）。
处置：WAV 注入/离线一律走整段预计算；麦克风流式路径 0.5s warm-up（报数不门控）。
**绝对不要**按 0.5s 短窗每窗重置 GRU（短窗陷阱，见 DESIGN.md 4.3）。

### 5.3 SAPI 合成音的两个坑

- **注册余量薄**：PVAD 在真实 ST-CMDS 语音上训练，对 SAPI TTS 声音（如 test_audio 的
  Huihui）目标检测偏弱——voice1b 全文仅 ~7 帧 P>0.5。真实麦克风语音余量更大。
- **伪说话人无效**：同一 SAPI 语音的 pitch ±25% / rate ±8 变体与本人 cos 仍 0.65–0.95，
  不能当 cohort/负样本用（会把 z 分数压垮）。cohort 必须用真实多人语料。

### 5.4 误打断（FAR）偏高

PVAD 帧级 FAR 0.276（干净）/0.526（增广），端到端误打断 7.0%/21.6%。缓解：
`--confirm 3`、`--pvad-threshold 0.6`、触发后全局冷却（需自行加）；根本改善靠训练侧
（硬负例、F1−λ·FAR 选模，见 TRAINING.md 第 8 节）。

### 5.5 ORT 告警 "Expected shape from model of {-1,137,3}"

pvad.onnx 导出时示例维度 137 留在了输出 shape 声明里，动态 T 实际正常。
CLI/demo 已把 pvad session 日志级别设为 ERROR 屏蔽；python 侧可忽略。
