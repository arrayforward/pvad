# -*- coding: utf-8 -*-
"""回归单格驱动：enroll 批量注册 -> double_voice --batch-list -> 判定汇总。

用法:
  python scripts/run_regression_cell.py --test-dir data/mixtures/test \
      --pvad-model models/pvad/pvad_v3.onnx --denoise rnnoise --max-n 200

判定与 scripts/eval_cpp_pvad.py / eval_pvad.py 的 judge() 一致:
  触发帧落在重叠区(±20 帧容忍)或触发帧标签为 2 = 正确; 未触发 = 漏打断; 否则 = 误打断。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOL = 20


def judge(trig, overlaps, labels):
    if trig is None or trig < 0:
        return "miss", None
    for st, ed in overlaps:
        if st - TOL <= trig <= ed + TOL:
            return "ok", trig - st
    if trig < len(labels) and labels[trig] == 2:
        return "ok", None
    return "false", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--pvad-model", default="models/pvad/pvad_v3.onnx")
    ap.add_argument("--denoise", default="rnnoise")
    ap.add_argument("--max-n", type=int, default=200)
    args = ap.parse_args()

    td = Path(args.test_dir)
    tag = td.name  # test / test (mixtures_v2 下也叫 test，用父目录区分)
    if td.parent.name != "data":
        tag = f"{td.parent.name}_{td.name}"
    tpl_dir = ROOT / "build" / "grid_tpl" / tag
    list_path = ROOT / "build" / f"grid_list_{tag}.tsv"
    out_path = ROOT / "build" / f"grid_out_{tag}.tsv"

    # 1) 批量注册（幂等：模板已存在则跳过整步）
    recs = []
    with open(td / "labels.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("overlap_frames"):
                recs.append(r)
    recs = recs[: args.max_n]
    if not tpl_dir.exists() or len(list(tpl_dir.glob("*.bin"))) < len(recs):
        print(f"[1/3] enroll batch -> {tpl_dir}")
        p = subprocess.run(
            [str(ROOT / "build" / "enroll.exe"), "--batch-jsonl",
             str(td / "labels.jsonl"), "--out-dir", str(tpl_dir),
             "--max-n", str(args.max_n), "--root", "."],
            cwd=ROOT, capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout[-2000:], p.stderr[-2000:])
            return 1
    else:
        print(f"[1/3] enroll batch cached ({len(list(tpl_dir.glob('*.bin')))} tpls)")

    # 2) 生成列表并跑批量管线
    with open(list_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(f"{r['path']}\tbuild/grid_tpl/{tag}/{r['id']}.bin\n")
    print(f"[2/3] double_voice batch ({len(recs)} files, model={args.pvad_model}, denoise={args.denoise})")
    p = subprocess.run(
        [str(ROOT / "build" / "double_voice.exe"), "--batch-list", str(list_path),
         "--batch-out", str(out_path), "--pvad-model", args.pvad_model,
         "--denoise", args.denoise, "--gate", "pvad"],
        cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-2000:])
        return 1

    # 3) 判定汇总
    results = {}
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            results[parts[0]] = (int(parts[1]), float(parts[2]))
    stats = {"miss": 0, "false": 0, "ok": 0}
    delays = []
    missing = 0
    for r in recs:
        key = r["path"]
        if key not in results:
            missing += 1
            stats["miss"] += 1
            continue
        trig, _ = results[key]
        verdict, delay = judge(trig, r["overlap_frames"], r["labels"])
        stats[verdict] += 1
        if delay is not None:
            delays.append(delay)
    n = sum(stats.values())
    dl = sum(delays) / max(len(delays), 1)
    print(f"[3/3] {args.test_dir} | model={Path(args.pvad_model).name} denoise={args.denoise} | "
          f"漏打断 {stats['miss'] / n:.3f} 误打断 {stats['false'] / n:.3f} "
          f"正确 {stats['ok'] / n:.3f} (n={n}, missing={missing}, 平均延迟 {dl:.1f} 帧)")
    print(f"GRIDROW\t{args.test_dir}\t{Path(args.pvad_model).name}\t{args.denoise}\t"
          f"{stats['miss'] / n:.3f}\t{stats['false'] / n:.3f}\t{stats['ok'] / n:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
