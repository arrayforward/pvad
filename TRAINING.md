# PVAD 训练文档（Personal VAD，目标说话人语音活动检测）

本文档覆盖 PVAD 模型的环境重建、数据管线、训练、评估与复现全流程。
C++ 侧集成见 [README.md](README.md)；生产模型约定见 `models/pvad/pvad*.md`。

当前模型版本：**v3（FiLM 条件，`models/pvad/pvad_v3.onnx`）为最新训练产物；
C++ 管线默认加载 `pvad.onnx`（v1）**，替换前请阅读下文「模型选择指南」。

---

## 1. 环境重建

### 系统依赖

- Windows + Git Bash（本文所有命令均在 Git Bash 下执行）
- Python 3.12（在 PATH 中）
- git + git-lfs（下载 ModelScope 数据集，均为 git-LFS 仓库）
- tar / unzip（解压语料）
- 无 GPU 要求，全部流程 CPU 可跑（实测 18 核）

### Python 环境

```bash
cd D:/ai-app/double-voice
python -m venv .venv
# torch 必须先装 CPU 版（PyPI 默认是 ~2.4GB 的 CUDA 捆绑版）
.venv/Scripts/python.exe -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

直接依赖（`requirements.txt`）：torch 2.13.0+cpu、onnxruntime 1.29、onnx 1.22、
onnxscript 0.7.1（torch.onnx 导出需要）、numpy 2.5、scipy 1.18、soundfile 0.14、
pyarrow 25（ST-CMDS parquet 解码）。国内网络建议 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

> 注意：`.venv` 曾被外部因素整个删除过一次（非脚本行为）。若脚本突然报
> "No such file or directory"，先检查 `.venv` 是否还在，重建只需重跑上面的命令。

---

## 2. 数据管线

### 2.1 语料来源

| 语料 | 规模 | 来源（实际使用） | 备注 |
|---|---|---|---|
| AISHELL-1 (slr33) | 400 人 / 178h | ModelScope `OmniData/AISHELL-1`（原始 tgz，git-LFS） | Apache-2.0 |
| ST-CMDS FreeST (slr38) | 443 人 / 110h | ModelScope `pengzhendong/free_st_chinese_mandarin_corpus`（parquet） | speaker 从文件名 `20170001P00338A0110.wav` 的 `Pxxxxx` 段解析 |
| Primewords (slr47) | 296 人 / 99h | ModelScope `manyeyes/Primewords-Chinese-Corpus-Set-1`（原始 tar.gz） | speaker 从 `set1_transcript.json` 的 `user_id` 解析（文件名是 UUID） |
| aidatatang_200zh (slr62) | — | **不可用**：OpenSLR 已下架（资源页 404），HF/MS 均无镜像 | 如需更多说话人可考虑 AISHELL-3/MagicData |
| MUSAN (slr17) | noise+music 1590 个 | ModelScope `manyeyes/MUSAN`（只取 `noise/`+`music/` 子集） | v2/v3 噪声增广 |
| RIR (slr28 系) | 60000 个 | ModelScope `manyeyes/rirs_noises`（`simulated_rirs/`） | v2/v3 混响增广 |

经验：openslr.org 直连 ~0.8MB/s 且镜像站（trmal/elda/magicdatatech）按 IP 限流到
~20KB/s；ModelScope git-LFS 国内 6-17MB/s，是首选渠道。原始压缩包保留在
`data/raw/`（含 `aug/` 下已解压的 MUSAN/RIR），清理磁盘时优先删别处。

### 2.2 整理与划分

```bash
# 解压语料到 data/extracted/<corpus>/ 后（脚本含各语料的 speaker 解析规则）
.venv/Scripts/python.exe scripts/prepare_corpus.py --corpus all --workers 8
# 产物: data/speakers/<corpus>/<speaker>/<utt>.wav (16k 单声道 PCM16, 过滤 <0.5s)
#       data/manifest.jsonl  {"corpus","speaker","path","duration_s"}
```

结果：**1139 说话人 / 294,892 条 / 388.1h**（aishell 400人/179.4h、
primewords 296人/99.0h、stcmds 443人/109.7h）。

说话人不重叠划分（`data/split.json`，种子 42）：train 950 / val 100 / test 89。
**所有评估都在 test 池上，训练/验证说话人与 test 零重叠。**

### 2.3 混合物合成（scripts/gen_mixtures.py）

每条样本：目标说话人语音 + 干扰说话人语音（不同人），目标:干扰 SNR ∈ [-5,10]dB，
3-8s，16kHz。标签为 10ms 帧级三分类（0 静音 / 1 非目标语音 / 2 目标语音），
由拼接位置精确生成（非 VAD 估计），重叠帧计为 2。每条记录 enrollment（目标说话人
3-10s 的其他句子，排除已混入源）和 `overlap_frames`（双讲区间，供端到端评估）。

三个数据版本：

- **v1 `data/mixtures/`**：干净混合。train 15000+4000、val/test 各 2000+500。
  后半为 **target-absent 负样本**（只有干扰人，enrollment 是不在场说话人，标签无 2）——
  无负样本时模型会学成"语音即目标"（FAR≈0.99），这是 v1 前期最重要的教训。
- **v2 `data/mixtures_v2/`**：同配方 + 增广：目标/干扰各自独立 70% 概率卷随机 RIR
  （RMS 归一化），60% 概率叠 MUSAN 噪声（SNR 5-20dB，噪声不计入任何标签类）。
- **v3 `data/mixtures_v3/`**：v1:v2 = 50:50 混合（`scripts/build_v3_data.py` 硬链接复用，
  零额外磁盘）+ 双干扰硬负例（train 700 / val·test 各 90，`--double-interferer`，
  两个干扰人同时说话、无目标、标签全 1）+ 50% 样本的 enrollment 卷 RIR 后重算
  embedding（注册语音混响漂移鲁棒）。

### 2.4 特征预计算（scripts/precompute_features.py）

对每个 split 目录预计算：
- `feats/<id>.npy`：80 维 log-mel fbank（**与 `src/fbank.cpp` 逐参数对齐**：
  25ms/10ms、fft512、预加重 0.97、对称 hamming、low 20Hz、log(max(e,1e-10))、
  无中心 padding）+ per-bin 均值归一化（3D-Speaker 做法）
- `emb/<id>.npy`：enrollment 经 `models/campplus.onnx`（onnxruntime）的 192 维
  embedding，L2 归一化

断点续跑（已存在则跳过），多进程。这些缓存是训练/评估的输入，**保留**；
重训同分布数据时可直接复用。

---

## 3. 模型与训练

### 3.1 架构演进

| | v1 / v2 | v3 |
|---|---|---|
| 条件机制 | emb broadcast 拼接到每帧（272 维输入） | **FiLM**：emb→每层 GRU 输入的 (γ,β) 逐维调制 `h⊙(1+γ)+β` |
| 网络 | GRU 2×128 + Linear | GRU 1×128 → FiLM → GRU 1×128 + Linear |
| 参数量 | 253,827 | 260,387 |

外部接口三版完全相同：`feats [B,T,80] + emb [B,192] → logits [B,T,3]`，
ONNX 可互相替换加载。

**为什么不是交叉注意力**：192 维单向量作单 token KV 时，softmax(1 token)≡1，
注意力输出退化为 V·emb（一个线性加性条件），严格弱于 FiLM 的乘性+加性调制。
`PvadModelAttn`（291K 参数）已在 `train_pvad.py --cond attn` 中实现但未训练；
若要让注意力真正生效，需要 enrollment 的多帧表示（CAM++ 中间层）而非单向量。

### 3.2 损失与关键实验依据

加权 CE（ignore padding）。最终权重 **[1, 2, 3]**（静音/非目标/目标），实验依据：

- 无负样本 + w=[1,1,4]：模型忽略 enrollment，随机 embedding 输出不变，FAR≈0.99
- 有负样本 + w=[1,1,4]：FAR 随训练反而恶化（ep1 0.50→ep2 0.53），过抑制优先
  被类不平衡（目标帧占 ~90%）放大
- 有负样本 + **w=[1,2,3]**：过抑制仍 3 倍优先，同时给干扰类 2 倍 FAR 压力，收敛最好
- 训练集截断（--max-train）曾踩坑：顺序切片会把追加在尾部的负样本全部切掉，
  已改为固定种子 shuffle 后截取

### 3.3 各版训练配置与耗时（CPU 18 核）

| | 数据 | epoch | 每 epoch | 总耗时 | 最优 ckpt |
|---|---|---|---|---|---|
| v1 | 12000（19000 中 shuffle 截取） | 8 | ~7-16min | ~1.2h | ep7（val F1 0.945） |
| v2 | 19000 全量增广 | 12 | ~11.5min | ~2.3h | ep12（val F1 0.907） |
| v3 | 19000 混合分布 | 15 | ~11-14min | ~3h | ep4（val F1 0.904） |

复训命令（以 v3 为例）：

```bash
.venv/Scripts/python.exe scripts/train_pvad.py --cond film --epochs 15 --batch 32 \
  --target-weight 3.0 --weight1 2.0 \
  --train-dir data/mixtures_v3/train --val-dir data/mixtures_v3/val \
  --ckpt-name best_v3.pt --log-name train_log_v3.json
```

---

## 4. 评估方法论

`scripts/eval_pvad.py`（v1/v2/v3 全部用同一份代码、同一迟滞参数）：

- **帧级指标**（`--frames`）：目标类(2) recall/precision/F1、FAR=P(判2|真1)、
  accuracy，按 SNR 分档（<0dB / 0-5dB / >5dB）
- **端到端双讲对照**（`--e2e`，各 500 条含双讲样本）：P(target) 过 confirm=2
  迟滞门控（>0.5 连续 2 帧触发，<0.3 清零）模拟打断判定；触发帧落在双讲区间
  （±20 帧容忍）为正确，无目标处触发为误打断，未触发为漏打断。对照组为
  CAM++ AS-norm 门控的 python 复现（500ms 窗、cohort 300、负模板=干扰人、
  z>3.0 + margin>0.15 + confirm=2）
- `--median 5`：帧级目标概率先做 5 帧中值滤波再入门控（结论：收益边际
  ~0.2-1pp，C++ 端可选）

### v1 / v2 / v3 三方对比（test 池 89 人，训练外）

帧级（ALL 档）：

| 测试集 | 模型 | recall | F1 | FAR | acc |
|---|---|---|---|---|---|
| v1 干净 | v1 | **0.976** | 0.936 | 0.350 | 0.901 |
| | v2 | 0.949 | 0.926 | 0.320 | 0.886 |
| | **v3** | 0.965 | **0.941** | **0.276** | **0.910** |
| v2 增广 | v1 | 0.951 | 0.876 | 0.683 | 0.779 |
| | v2 | 0.967 | **0.901** | 0.555 | 0.836 |
| | **v3** | 0.958 | 0.900 | **0.526** | **0.834** |

端到端双讲（漏打断 / **误打断** / 正确率）：

| 测试集 | v1 | v2 | v3 | AS-norm 对照 |
|---|---|---|---|---|
| v1 干净 | 0.6% / 8.8% / 90.6% | 0.4% / 14.6% / 85.0% | **0.6% / 7.0% / 92.4%** | 64.8% / 0% / 35.2% |
| v2 增广 | 0.8% / 26.8% / 72.4% | **0.2% / 18.0% / 81.8%** | 0.2% / 21.6% / 78.2% | 74.0% / 0% / 26.0% |

（v3 + 5 帧中值滤波：干净 6.6%，增广 20.4%。）

---

## 5. 模型选择指南

| 场景 | 推荐 | 依据 |
|---|---|---|
| 干净近场（当前默认生产环境） | **v3**（`pvad_v3.onnx`） | 干净条件全面最优：误打断 7.0%、正确率 92.4%、FAR 0.276，且帧级指标两头都不垫底 |
| 强噪声/混响为主 | v2（`pvad_v2.onnx`） | 增广条件误打断 18.0% 仍最低；v3 在增广 e2e 上 21.6% 略输 |
| 无训练条件/基线对照 | v1（`pvad.onnx`） | 干净条件第二，架构最简单 |

**已知问题——checkpoint 选择准则**：训练按 val F1 存最优，F1 偏 recall。
v3 的 ep12 val FAR 0.317 明显优于 ep4 的 0.440，但 F1 略低而未被保存——
增广 e2e 未达标（21.6% vs 18%）部分源于此。下一步应改为 **F1−λ·FAR 联合准则**
（预计零训练成本即可把增广误打断压到达标线）。

---

## 6. 复现清单（从原始压缩包到 pvad_v3.onnx）

```bash
# 0. 环境（见第 1 节）；原始数据在 data/raw/（ModelScope 仓库，git lfs pull）
# 1. 语料整理（如需从头：先解压 data/raw 压缩包到 data/extracted/）
.venv/Scripts/python.exe scripts/prepare_corpus.py --corpus all --workers 8
# 2. 说话人划分（已有 data/split.json 可跳过）
# 3. v1 混合物（干净）
.venv/Scripts/python.exe scripts/gen_mixtures.py --n 15000 --negatives 4000 --pool train --seed 3003 --out data/mixtures/train
.venv/Scripts/python.exe scripts/gen_mixtures.py --n 2000  --negatives 500  --pool val   --seed 1001 --out data/mixtures/val
.venv/Scripts/python.exe scripts/gen_mixtures.py --n 2000  --negatives 500  --pool test  --seed 2002 --out data/mixtures/test
# 4. v2 混合物（增广；AUG="--rir-dir data/raw/aug/RIRS_NOISES/simulated_rirs --noise-dir data/raw/aug/musan"）
#    同第 3 步命令加 $AUG，输出到 data/mixtures_v2/{train,val,test}
# 5. 各版本 precompute
.venv/Scripts/python.exe scripts/precompute_features.py --dir data/mixtures/train --workers 12   # val/test 同理
# 6. v3 数据（v1+v2 混合 + enrollment RIR）
.venv/Scripts/python.exe scripts/build_v3_data.py
#    双干扰硬负例（train 700 / val·test 各 90，一半干净一半增广）
.venv/Scripts/python.exe scripts/gen_mixtures.py --n 0 --negatives 350 --double-interferer --pool train --seed 3003 --out data/mixtures_v3/train --append
.venv/Scripts/python.exe scripts/gen_mixtures.py --n 0 --negatives 350 --double-interferer --pool train --seed 3004 --out data/mixtures_v3/train --append --rir-dir ... --noise-dir ...
.venv/Scripts/python.exe scripts/precompute_features.py --dir data/mixtures_v3/train --workers 12  # val/test 同理
# 7. 训练
.venv/Scripts/python.exe scripts/train_pvad.py --cond film --epochs 15 --batch 32 \
  --target-weight 3.0 --weight1 2.0 \
  --train-dir data/mixtures_v3/train --val-dir data/mixtures_v3/val \
  --ckpt-name best_v3.pt --log-name train_log_v3.json
# 8. 评估
.venv/Scripts/python.exe scripts/eval_pvad.py --all --ckpt models/pvad/best_v3.pt --test-dir data/mixtures/test
.venv/Scripts/python.exe scripts/eval_pvad.py --all --ckpt models/pvad/best_v3.pt --test-dir data/mixtures_v2/test
# 9. 导出（一致性自动验证，误差 <1e-4）
.venv/Scripts/python.exe scripts/export_onnx.py --ckpt models/pvad/best_v3.pt --out models/pvad/pvad_v3.onnx
```

ONNX 导出一个坑：FiLM 的 `chunk(2)` 经 torch.onnx 版本转换为 opset 17 后
Split 节点带非法 `num_outputs` 属性，需改用切片（已在代码中修复）。

---

## 7. 磁盘占用与产物说明

| 目录 | 大小 | 内容 | 保留策略 |
|---|---|---|---|
| `data/speakers/` | 43G | 整理后的 16k 单声道语料（1139 人/388h） | 保留（一切数据的源头） |
| `data/mixtures/` | 8.2G | v1 干净混合物 + labels + feats/emb 缓存 | 保留（v1 评估基线、v3 复用） |
| `data/mixtures_v2/` | 8.1G | v2 增广混合物（同构） | 保留（同上） |
| `data/mixtures_v3/` | 408M | v3 自有文件（大部分为硬链接，不占额外空间） | 保留 |
| `data/raw/` | ~54G | 原始压缩包（ms_aishell1 15G / ms_musan ~9G / ms_primewords 8.5G / ms_stcmds 11G / ms_rir ~1G）+ `aug/` 已解压增广库 8.9G | 保留压缩包（之前决策）；`.git/lfs` 缓存已清理 |
| `models/` | 36M | campplus.onnx、silero_vad.onnx、pvad/ 全部版本 | 保留，不覆盖旧版本 |
| `.venv/` | ~2G | Python 训练环境（可由 requirements.txt 重建） | 可删可重建 |
| `data/manifest.jsonl` / `split.json` | <100M | 语料清单 / 说话人划分 | 保留 |

D: 盘当前剩余 ~85GB。若需释放空间，优先顺序：`data/raw` 压缩包（可从
ModelScope 重下，~54G）> `data/mixtures_v2`（可重新合成）> 其余勿动。
`data/extracted/`（解压中间产物，~25G）已在早前清理，需要时从 `data/raw`
重新解压即可。

## 8. 已知问题与下一步

1. **checkpoint 选择准则**：val F1 偏 recall，改 F1−λ·FAR 联合选模（最高优先级，
   零训练成本）
2. **硬负例扩充**：双干扰样本扩到 2000+ 条，并加入"干扰人音色接近目标"的配对
   （按 enrollment embedding 近邻挑选）
3. **enrollment 多帧表示**：当前 192 维单向量使交叉注意力退化；取 CAM++ 中间层
   多帧表示后注意力条件才有意义，`--cond attn` 已就绪
4. **增广 e2e 差距**：v3 增广误打断 21.6% 未达 ≤18% 目标（帧级 FAR 已最优），
   上述 1+2 预计可闭合
5. aidatatang_200zh（600 人）OpenSLR 已下架，如说话人多样性成为瓶颈可补
   AISHELL-3 / MagicData（ModelScope 均有镜像）

---

## 9. v4（2026-08-26，当前最新）

**目标**：干净和增广两条件同时打平/超过历史最优（验收线：干净正确率 ≥92.4% 且
增广 ≥81.8%，核心看误打断/FAR）。

### 数据 v4（`data/mixtures_v4/`）
- 主体硬链接复用 v3 的 50/50 干净/增广分布（含双干扰硬负例 700 条）
- **新增 2000 条音色相近易混淆负样本**（`scripts/build_speaker_centroids.py` 给
  1139 个说话人算 CAM++ 质心 + top5 近邻，`gen_mixtures.py --confusable-file`：
  负样本干扰人改为目标说话人 embedding 最近 top5 之一，一半干净一半增广）
- 负样本占比调回 25%（19263 条：易混淆 2000 + 双干扰 700 + 随机 2116）
- 测试集不新建，直接用 `data/mixtures/test` 和 `data/mixtures_v2/test` 保证历史可比

### 训练与选模
FiLM 架构不变（260,387 参数）、超参同 v3（[1,2,3]、batch32、lr1e-3）、15 epoch、
**每 epoch 全存 checkpoint**（`best_v4_epNN.pt`）。

选模过程（只用 val）：F1−λ·FAR 在 λ=1 与 λ=2 下**都指向 ep14**（val F1 0.859/
FAR 0.349），但 ep14 的 **val e2e 反而更差**（误打断 14.2% vs ep11 的 11.2%）——
帧级 FAR 与样本级误打断再次背离（ep14 recall 掉到 0.879 换 FAR 0.157）。
最终按 val e2e 选 **ep11**（val F1 0.882/FAR 0.430）。教训：F1−λ·FAR 联合准则
方向正确但不够，**val e2e 必须参与终选**；λ 在本数据上无法区分候选点。

### 结果（v1/v2/v3/v4 四方对比，python 口径，各 500 条）

端到端（漏打断 / **误打断** / 正确率）：

| 测试集 | v1 | v2 | v3 | **v4** |
|---|---|---|---|---|
| v1 干净 | 0.6% / 8.8% / 90.6% | 0.4% / 14.6% / 85.0% | 0.6% / 7.0% / 92.4% | **1.0% / 5.0% / 94.0%** |
| v2 增广 | 0.8% / 26.8% / 72.4% | 0.2% / 18.0% / 81.8% | 0.2% / 21.6% / 78.2% | **0.6% / 16.4% / 83.0%** |

帧级 FAR / recall（ALL 档）：v4 干净 **0.210** / 0.940，增广 **0.446** / 0.926
（FAR 两条件均为历代最低；recall 较 v1 的 0.976 低 3.6pp，主要换误打断下降）。

C++ 口径（pvad+rnnoise，各 200 条）：v4 干净 **97.5%**（误 1.5%）、
增广 **82.5%**（误 16.0%），均超 v3 的 95.5%/79.0%。

### 验收结论：**双验收线均达标**
- 干净 94.0% ≥ 92.4% ✓（误打断 5.0% 历代最低）
- 增广 83.0% ≥ 81.8% ✓（误打断 16.4% 历代最低，首次超过 v2）
- 代价：recall 略降（漏打断 1.0%/0.6%，仍在个位数）

复训/评估命令：`train_pvad.py --cond film --save-all-epochs ...`（同第 6 节，
数据换 `data/mixtures_v4/`）；易混淆负样本见 `scripts/build_speaker_centroids.py`
与 `scripts/build_v4_data.py`。

---

## 10. 流式导出（pvad_v4_stream.onnx，2026-08-27）

### 方法
GRU 隐状态外置（`scripts/export_stream_onnx.py`）：forward 签名
`(feats_chunk[B,t,80], emb[B,192], h0[2,B,128]) → (logits[B,t,3], hN[2,B,128])`，
初始 h0=0，chunk 间 state 串接。CMVN 不进 ONNX，由调用方负责。
opset 17。chunk=1/5/50 与整段数学等价（2.62e-06，数值噪声级）。

### CMVN 流式化：五方案实测与最终选择（关键风险点）
整段 per-bin 均值归一化需要未来信息，流式只能因果近似。各 200 条 e2e
（confirm=2、±20 帧）对整段基线（干净 95.0% / 增广 83.0%）的正确率：

| 方案 | 干净 | 增广 | 结论 |
|---|---|---|---|
| running（累积均值） | 20.5% | 36.0% | 灾难：均值未收敛期全漏 |
| sliding 3s | 19.5% | 39.0% | 同上 |
| EMA α=0.02 | 82.5% | 89.0% | 最好但干净 -12.5pp，不达标 |
| EMA α=0.05 / 0.1 | 74.0% / 56.5% | 82.0% / 62.5% | 更快收敛反而更差 |
| running+enrollment 先验 | 69.5% | 67.5% | 先验把均值拉离混合分布，有害 |
| **EMA α=0.02 + EMA 特征微调（最终）** | **94.5%** | **82.5%** | **两条件均 -0.5pp，达标** |

结论：**CMVN 技巧无法绕过训练/部署不一致**，正解是让训练分布与部署一致——
从 best_v4.pt 以 EMA-CMVN 特征（α=0.02）微调 4 epoch（lr 1e-4，
`train_pvad.py --feats-subdir feats_ema --init-from`，checkpoint best_v4s.pt），
再从微调权重导出流式 ONNX。注意：此时流式模型与离线整段模型（pvad_v4.onnx）
已是不同权重，离线仍用 v4 整段模型，实时用 stream 模型，两者各司其职。

### warm-up 建议
EMA 时间常数 50 帧（0.5s）。微调后模型对早期未收敛统计鲁棒，
3-8s 短片段（语音从头开始的最坏情况）退化仅 0.5pp；C++ 侧建议：
会话起始 0.5s 内不做门控判定（或接受该窗口内略高的漏检）。

### 流式 vs 整段 e2e（EMA α=0.02, chunk=5, 各 200 条）

| 条件 | 整段基线 | 流式（微调后） | 退化 |
|---|---|---|---|
| v1 干净 | 95.0% | **94.5%**（漏 1.0%/误 4.5%） | -0.5pp ✓ |
| v2 增广 | 83.0% | **82.5%**（漏 0.0%/误 17.5%） | -0.5pp ✓ |

交付：`models/pvad/pvad_v4_stream.onnx` + `pvad_v4_stream.md`（接口与 CMVN 约定）、
`scripts/export_stream_onnx.py`、`scripts/eval_stream.py`（CMVN 分析 + 流式 e2e）、
`models/pvad/best_v4s.pt`（EMA 微调 checkpoint）+ `train_log_v4s.json`。

---

## 11. v5（2026-08-28）：enrollment 多帧表示 + 真交叉注意力

### 动机与架构
v4 分析过"单向量作单 token KV 的注意力数学退化"。v5 把 enrollment 按 1s 切子段
（整 1s 切分、尾段丢弃，N=3-10），每段过 CAM++ → tokens [N,192]
（`precompute_features.py --tokens` → `emb_tokens/`）。帧特征投影 (80→128) 作 query、
tokens 作 K/V 的**手写 2 头交叉注意力**（`CrossAttn`，残差融合）进 GRU。
消融两版：纯注意力 `attn5`（291,331 参数）、注意力+FiLM `film_attn`（390,147 参数，
FiLM 用 mask 均值池化 tokens，保留 v4 的 gru1/gru2 结构便于初始化）。
ONNX 导出注意：nn.MultiheadAttention 和 .chunk() 在 opset 17 版本转换下都有坑
（Split num_outputs），需手写注意力 + 切片（教训同 v3）。

### 训练与选模
数据复用 mixtures_v4 分布，两版均从 best_v4.pt 初始化 GRU/输出层（attn5 需
gru1/gru2→单层 GRU 键重映射；形状不匹配的键随机初始化），12000 条 × 10 epoch
（~7-9min/epoch），权重 [1,2,3]，每 epoch 存 ckpt，**val e2e 选模**。

### 消融结果（val e2e → test e2e）

| 版本 | val e2e 最优 | test 干净 | test 增广 |
|---|---|---|---|
| film_attn ep08 | **89.6%**（误 9.0%） | **93.0%**（误 5.6%） | **89.4%**（误 10.0%） |
| attn5 ep10 | 83.6%（误 16.2%） | 89.2%（误 10.4%） | 80.0%（误 19.6%） |

**注意力+FiLM 显著优于纯注意力**；纯注意力甚至不如 v4 的纯 FiLM。

### v1/v3/v4/v5 对比（test e2e，各 500 条，同口径）

| 条件 | v1 | v3 | v4 | v5(film_attn) |
|---|---|---|---|---|
| 干净正确率 | 90.6% | 92.4% | 94.0% | **93.0%**（误 5.6%） |
| 增广正确率 | 72.4% | 78.2% | 83.0% | **89.4%**（误 **10.0%**） |

### 验收：**达标**
- 增广 89.4% ≥ 84% ✓（主攻目标：误打断 16.4%→10.0%，-6.4pp）
- 干净 93.0% ≥ 93% ✓（踩线，未超 v4 的 94.0% 但在 -1pp 允许带内）

### 注意力行为分析（如实）
抽取注意力权重：max-attn 在目标帧（0.72）与干扰帧（0.64）、负样本（0.58-0.75）
之间没有清晰分离——**注意力权重不构成可解释的"匹配 enrollment"证据**
（N 小、softmax 恒集中于某 token）。v5 的收益更可能来自"多帧 enrollment 提供的
更丰富条件表示"整体（注意力输出 + 池化 FiLM 共同作用），而非可解释的检索式匹配。

### 流式版（pvad_v5_stream.onnx）
chunk 对齐 5.19e-06 ✓；但 EMA-CMVN 微调（同 v4_stream 流程）后流式 e2e：
干净 91.5%（-2.0pp 踩线）、**增广 71.0%（-17pp，不达标，误打断 12%→29%）**。
注意力条件路径对 EMA 统计漂移远比 FiLM 敏感。**实时生产暂留 pvad_v4_stream.onnx**；
v5 流式需更长 EMA 微调或注意力输入归一化改造（后续工作）。

### 交付
`models/pvad/pvad_v5.onnx`（parity 5.25e-06）+ `pvad_v5.md`、
`pvad_v5_stream.onnx` + `pvad_v5_stream.md`（含不达标标注）、
`best_v5fa.pt`(=ep08)/`best_v5a.pt`(=ep10)/`best_v5s.pt` + 各 train_log、
`scripts/`（CrossAttn、tokens 预计算、v5 流式 wrapper、eval_stream v5 支持）。
v1-v4 产物未动。
