# -*- coding: utf-8 -*-
"""构建 v4 训练数据: 硬链接复用 v3 全部样本, 预留易混淆负样本配额。

用法: python scripts/build_v4_data.py
输出: data/mixtures_v4/{train,val}/labels.jsonl + 硬链接 wav/feats/emb
之后运行 gen_mixtures --confusable-file ... --append 补易混淆负样本, 再 precompute。
"""
import json
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvad_common import ROOT, load_labels  # noqa: E402

# split -> (从 v3 取的条数, 预留易混淆负样本数)
PLAN = {"train": (19000, 2000), "val": (2350, 150)}


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    rng = random.Random(888)
    for split, (n_base, n_conf) in PLAN.items():
        src_dir = ROOT / "data" / "mixtures_v3" / split
        out_dir = ROOT / "data" / "mixtures_v4" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "feats").mkdir(exist_ok=True)
        (out_dir / "emb").mkdir(exist_ok=True)
        recs = load_labels(src_dir / "labels.jsonl")
        rng.shuffle(recs)
        recs = recs[:n_base]
        for r in recs:
            link(ROOT / r["path"], out_dir / f"{r['id']}.wav")
            link(src_dir / "feats" / f"{r['id']}.npy",
                 out_dir / "feats" / f"{r['id']}.npy")
            link(src_dir / "emb" / f"{r['id']}.npy",
                 out_dir / "emb" / f"{r['id']}.npy")
        with open(out_dir / "labels.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_neg = sum(1 for r in recs if r.get("negative"))
        print(f"[{split}] base {len(recs)} (负样本 {n_neg}), "
              f"预留易混淆负样本 {n_conf}")
    print("完成。接下来: gen_mixtures --confusable-file data/speaker_top5.json "
          "--append 补易混淆负样本 (一半干净一半增广), 再 precompute_features。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
