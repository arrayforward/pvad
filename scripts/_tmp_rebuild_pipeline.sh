#!/usr/bin/env bash
# 重建被删数据 (pvad-data 下): speakers -> mixtures v1/v2/v3/v4/long
# 断点续跑: prepare_corpus 跳过已存在; gen_mixtures 重写 labels 但幂等
set -e
cd /d/ai-app/double-voice
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
D=/d/ai-app/pvad-data
RAW=$D/raw
EX=$D/_tmp_extract

echo "== 1. 解压 aishell + primewords 到 _tmp_extract =="
mkdir -p $EX/aishell $EX/primewords
[ -d $EX/aishell/data_aishell/wav ] || tar -xzf $RAW/ms_aishell1/raw/33/data_aishell.tgz -C $EX/aishell
# aishell 二级 speaker 包
if [ ! -d $EX/aishell/data_aishell/wav/S0002 ]; then
  for f in $EX/aishell/data_aishell/wav/S*.tar.gz; do
    d="${f%.tar.gz}"; mkdir -p "$d" && tar -xzf "$f" -C "$d" && rm "$f"
  done
fi
[ -d $EX/primewords/primewords_md_2018_set1 ] || tar -xzf $RAW/ms_primewords/primewords_md_2018_set1.tar.gz -C $EX/primewords

echo "== 2. prepare_corpus (speakers/ 重建) =="
$PY scripts/prepare_corpus.py --corpus all --workers 8

echo "== 3. 删除 _tmp_extract (AGENTS.md 允许的 _tmp 前缀) =="
rm -rf $D/_tmp_extract

echo "== 4. v1 干净混合物 =="
AUG="--rir-dir $RAW/aug/RIRS_NOISES/simulated_rirs --noise-dir $RAW/aug/musan"
$PY scripts/gen_mixtures.py --n 15000 --negatives 4000 --pool train --seed 3003 --out $D/mixtures/train
$PY scripts/gen_mixtures.py --n 2000 --negatives 500 --pool val --seed 1001 --out $D/mixtures/val
$PY scripts/gen_mixtures.py --n 2000 --negatives 500 --pool test --seed 2002 --out $D/mixtures/test

echo "== 5. v2 增广 val/test (train 太大先不重建, 可随时再生) =="
$PY scripts/gen_mixtures.py --n 2000 --negatives 500 --pool val --seed 1001 --out $D/mixtures_v2/val $AUG
$PY scripts/gen_mixtures.py --n 2000 --negatives 500 --pool test --seed 2002 --out $D/mixtures_v2/test $AUG

echo "== 6. 预计算 v1/v2 =="
for d in mixtures/train mixtures/val mixtures/test mixtures_v2/val mixtures_v2/test; do
  $PY scripts/precompute_features.py --dir $D/$d --workers 12
done

echo "== 7. v3 构建 + 硬负例 =="
$PY scripts/build_v3_data.py
for sp in train:700:3003 val:90:1001 test:90:2002; do
  IFS=: read s n seed <<< "$sp"
  $PY scripts/gen_mixtures.py --n 0 --negatives $((n/2)) --double-interferer --pool $s --seed $seed --out $D/mixtures_v3/$s --append
  $PY scripts/gen_mixtures.py --n 0 --negatives $((n-n/2)) --double-interferer --pool $s --seed $((seed+1)) --out $D/mixtures_v3/$s --append $AUG
done
for sp in train val test; do $PY scripts/precompute_features.py --dir $D/mixtures_v3/$sp --workers 12; done

echo "== 8. v4 构建 + 易混淆硬负例 =="
$PY scripts/build_v4_data.py
for sp in train:2000 val:150; do
  IFS=: read s n <<< "$sp"
  half=$((n/2))
  $PY scripts/gen_mixtures.py --n 0 --negatives $half --confusable-file $D/speaker_top5.json --pool $s --seed 4001 --out $D/mixtures_v4/$s --append
  $PY scripts/gen_mixtures.py --n 0 --negatives $((n-half)) --confusable-file $D/speaker_top5.json --pool $s --seed 4002 --out $D/mixtures_v4/$s --append $AUG
done
for sp in train val; do $PY scripts/precompute_features.py --dir $D/mixtures_v4/$sp --workers 12; done
# v4 负样本占比调回 25% (与历史 train_log_v4 一致的比例重调)
$PY - <<'PYEOF'
import json, random
p = "/d/ai-app/pvad-data/mixtures_v4/train/labels.jsonl"
recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
pos = [r for r in recs if not r.get("negative")]
double = [r for r in recs if r.get("negative") and r.get("double_interferer")]
confusable = [r for r in recs if r.get("negative") and not r.get("double_interferer")
              and r["id"].startswith("mix_") and int(r["id"].split("_")[-1]) >= 19000]
rand_neg = [r for r in recs if r.get("negative") and not r.get("double_interferer")
            and not (r["id"].startswith("mix_") and int(r["id"].split("_")[-1]) >= 19000)]
rng = random.Random(999); rng.shuffle(rand_neg)
keep = rand_neg[:2116]
out = pos + double + confusable + keep
rng.shuffle(out)
with open(p, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
n_neg = sum(1 for r in out if r.get("negative"))
print(f"v4 train: {len(out)} 条, 负样本 {n_neg} ({n_neg/len(out):.1%})")
PYEOF

echo "== 9. 长流重建 (3500/300/300) =="
$PY scripts/gen_longstream.py --split train --n 3500 --seed 7001
$PY scripts/gen_longstream.py --split val --n 300 --seed 7002
$PY scripts/gen_longstream.py --split test --n 300 --seed 7003
for d in train val test; do $PY scripts/precompute_features.py --dir $D/mixtures_long/$d --cmvn ema --workers 12; done

echo "== 10. 回归验证 (v4+rnnoise 干净 200 条, 期望 ≈97.5%) =="
$PY scripts/run_regression_cell.py --test-dir $D/mixtures/test --pvad-model models/pvad/pvad_v4.onnx --denoise rnnoise --max-n 200

echo "== 11. 长流验证 =="
$PY scripts/eval_longstream.py --stream-onnx models/pvad/pvad_v4_stream.onnx --split test --max-n 300

echo "== 12. manifest 统计 =="
$PY -c "
import json
from collections import defaultdict
spk, cnt, dur = defaultdict(set), defaultdict(int), defaultdict(float)
for line in open('$D/manifest.jsonl', encoding='utf-8'):
    r = json.loads(line)
    spk[r['corpus']].add(r['speaker']); cnt[r['corpus']] += 1; dur[r['corpus']] += r['duration_s']
for c in sorted(cnt):
    print(f'{c}: {len(spk[c])} 人 / {cnt[c]} 条 / {dur[c]/3600:.1f}h')
"
echo REBUILD_ALL_DONE
