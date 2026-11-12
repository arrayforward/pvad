#!/usr/bin/env bash
# 重建管线第二段: speakers/ 已重建完成 (1139人/294892条/388.1h, 与历史一致),
# _tmp_extract 已删。本脚本从混合物合成开始。
# 第一段见 _tmp_rebuild_pipeline.sh (步骤 1-3 已执行)。
set -e
cd /d/ai-app/double-voice
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
D=/d/ai-app/pvad-data
RAW=$D/raw
AUG="--rir-dir $RAW/aug/RIRS_NOISES/simulated_rirs --noise-dir $RAW/aug/musan"

echo "== 4. v1 干净混合物 (train 15000+4000 / val 2000+500 / test 2000+500) =="
$PY scripts/gen_mixtures.py --n 15000 --negatives 4000 --pool train --seed 3003 --out $D/mixtures/train
$PY scripts/gen_mixtures.py --n 2000 --negatives 500 --pool val --seed 1001 --out $D/mixtures/val
$PY scripts/gen_mixtures.py --n 2000 --negatives 500 --pool test --seed 2002 --out $D/mixtures/test

echo "== 5. v2 增广混合物 (全量, 同种子) =="
$PY scripts/gen_mixtures.py --n 15000 --negatives 4000 --pool train --seed 3003 --out $D/mixtures_v2/train $AUG
$PY scripts/gen_mixtures.py --n 2000 --negatives 500 --pool val --seed 1001 --out $D/mixtures_v2/val $AUG
$PY scripts/gen_mixtures.py --n 2000 --negatives 500 --pool test --seed 2002 --out $D/mixtures_v2/test $AUG

echo "== 6. 预计算 v1/v2 =="
for d in mixtures/train mixtures/val mixtures/test mixtures_v2/train mixtures_v2/val mixtures_v2/test; do
  $PY scripts/precompute_features.py --dir $D/$d --workers 12
done

echo "== 7. v3 构建 + 双干扰硬负例 =="
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
# v4 负样本占比调回 25% (易混淆 id >= mix_019000, 见 build_v4 PLAN)
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

echo REBUILD_ALL_DONE
