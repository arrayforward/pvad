#!/usr/bin/env bash
# 重建管线第三段: pipeline2 在 v4 负样本重调处中断 (heredoc 内 MSYS 路径 Windows python 不认)。
# v1/v2/v3 全量 + v4 构建/易混淆/预计算已完成; 本脚本从重调开始。
set -e
cd /d/ai-app/double-voice
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
D=/d/ai-app/pvad-data

echo "== 8b. v4 负样本占比调回 25% =="
$PY - <<'PYEOF'
import json, random
p = r"D:/ai-app/pvad-data/mixtures_v4/train/labels.jsonl"
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
print(f"pos={len(pos)} double={len(double)} confusable={len(confusable)} rand_neg_pool={len(rand_neg)} keep={len(keep)}")
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
