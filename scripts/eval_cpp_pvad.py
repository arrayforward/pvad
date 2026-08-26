# -*- coding: utf-8 -*-
"""C++ double_voice (pvad 门控) 端到端双讲评估，对照 scripts/eval_pvad.py 的 python 路径。

对 data/mixtures/test 里前 N 条含双讲区间的样本:
  1) enroll 用该样本的 enrollment wav 现场注册模板 (走真实产品路径)
  2) double_voice --wav mix --template tpl --gate pvad 离线跑, 取第一个 INTERRUPT 帧
  3) 判定逻辑与 eval_pvad.py 的 judge() 相同:
     触发帧落在重叠区(±20 帧容忍)或触发帧标签为 2 = 正确; 未触发 = 漏打断; 否则 = 误打断

用法: python scripts/eval_cpp_pvad.py [--max-n 50]
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOL = 20


def judge(trig, overlaps, labels):
    if trig is None:
        return "miss", None
    for st, ed in overlaps:
        if st - TOL <= trig <= ed + TOL:
            return "ok", trig - st
    if trig < len(labels) and labels[trig] == 2:
        return "ok", None
    return "false", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=50)
    ap.add_argument("--test-dir", default=str(ROOT / "data" / "mixtures" / "test"))
    ap.add_argument("--denoise", default=None, choices=[None, "off", "rnnoise"],
                    help="透传给 double_voice 的 --denoise（默认不传=off）")
    ap.add_argument("--pvad-model", default=None)
    args = ap.parse_args()

    recs = []
    with open(Path(args.test_dir) / "labels.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("overlap_frames"):
                recs.append(r)
    recs = recs[: args.max_n]
    extra = []
    if args.denoise:
        extra += ["--denoise", args.denoise]
    if args.pvad_model:
        extra += ["--pvad-model", args.pvad_model]
    print(f"双讲样本 {len(recs)} 条 (test_dir={args.test_dir}, extra={extra})")

    stats = {"miss": 0, "false": 0, "ok": 0}
    delays = []
    with tempfile.TemporaryDirectory() as td:
        tpl = Path(td) / "tpl.bin"
        for i, r in enumerate(recs):
            mix = ROOT / r["path"]
            enroll_wav = ROOT / r["enrollment"]
            p = subprocess.run(
                [str(ROOT / "build" / "enroll.exe"), "--out", str(tpl),
                 "--pos", str(enroll_wav)],
                cwd=ROOT, capture_output=True, text=True)
            if p.returncode != 0:
                print(f"[{r['id']}] enroll 失败: {p.stderr.strip()}")
                stats["miss"] += 1
                continue
            p = subprocess.run(
                [str(ROOT / "build" / "double_voice.exe"), "--wav", str(mix),
                 "--template", str(tpl), "--gate", "pvad", *extra],
                cwd=ROOT, capture_output=True, text=True)
            m = re.search(r"\[t=\s*([0-9.]+)\].*>>> INTERRUPT <<<", p.stdout)
            trig = int(round(float(m.group(1)) * 100)) if m else None
            verdict, delay = judge(trig, r["overlap_frames"], r["labels"])
            stats[verdict] += 1
            if delay is not None:
                delays.append(delay)
            if (i + 1) % 10 == 0:
                print(f"  进度 {i + 1}/{len(recs)} "
                      f"(ok={stats['ok']} miss={stats['miss']} false={stats['false']})")

    n = sum(stats.values())
    print(f"\n=== C++ pvad 门控 端到端 ({n} 条) ===")
    print(f"漏打断 {stats['miss'] / n:.3f}  误打断 {stats['false'] / n:.3f}  "
          f"正确 {stats['ok'] / n:.3f}  平均延迟 {sum(delays) / max(len(delays), 1):.1f} 帧")
    return 0


if __name__ == "__main__":
    sys.exit(main())
