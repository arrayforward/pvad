# -*- coding: utf-8 -*-
"""真人集离线调参：从 eval_realhuman 落盘的 ident/conf/p2 序列扫阈值网格。

不重跑模型，直接对每个候选工作点重算判定：
  direct_v3/v4: closed = (ident < TH) and (conf >= UNC)  →  open = not closed
  v7b:          gate_trigger(p2, thr, hyst, confirm) 是否触发

报告每个模型的：均衡点（Youden J = A_mix放行 + B_only关闭 - 1 最大）、
A_mix≥90% / ≥95% 约束下 B_only 关闭最大的工作点。

用法: python scripts/sweep_realhuman.py [--rows build/eval_realhuman_full.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

DV = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DV / "scripts"))
from compare_tse_pvad import gate_trigger  # noqa: E402


def direct_metrics(rows, th, unc, prefix):
    out = {}
    for sc in ("A_mix", "B_only", "A_only"):
        sub = [r for r in rows if r["scenario"] == sc]
        open_n = sum(1 for r in sub
                     if not (r[f"{prefix}_ident"] < th and r[f"{prefix}_conf"] >= unc))
        out[sc] = open_n / max(1, len(sub))
    return out


def v7b_metrics(rows, thr, hyst, confirm):
    out = {}
    for sc in ("A_mix", "B_only", "A_only"):
        sub = [r for r in rows if r["scenario"] == sc]
        open_n = sum(1 for r in sub
                     if gate_trigger(np.asarray(r["v7b_p2"]), thr, hyst, confirm)
                     is not None)
        out[sc] = open_n / max(1, len(sub))
    return out


def pick_points(grid):
    """grid: [(params, m)] -> dict 均衡/90约束/95约束。"""
    def b_close(m):
        return 1 - m["B_only"]
    best_j = max(grid, key=lambda g: g[1]["A_mix"] + b_close(g[1]) - 1)
    ok90 = [g for g in grid if g[1]["A_mix"] >= 0.90]
    ok95 = [g for g in grid if g[1]["A_mix"] >= 0.95]
    best90 = max(ok90, key=lambda g: b_close(g[1])) if ok90 else None
    best95 = max(ok95, key=lambda g: b_close(g[1])) if ok95 else None
    return {"balanced": best_j, "a90": best90, "a95": best95}


def fmt(point):
    if point is None:
        return "无解"
    p, m = point
    return (f"{p} -> A_mix {m['A_mix']*100:.1f}% / A_only {m['A_only']*100:.1f}% / "
            f"B关闭 {(1-m['B_only'])*100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(DV / "build" / "eval_realhuman_full.json"))
    args = ap.parse_args()
    with open(args.rows, encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    print(f"样本 {len(rows)} 行 (A_mix/B_only/A_only 各 "
          f"{sum(1 for r in rows if r['scenario']=='A_mix')})")

    result = {}
    for prefix, name in (("v3", "direct_v3"), ("v4", "direct_v4")):
        grid = []
        for unc in (0.0, 0.1, 0.2, 0.3):
            for th in np.arange(0.01, 0.61, 0.01):
                m = direct_metrics(rows, round(float(th), 2), unc, prefix)
                grid.append(({"TH": round(float(th), 2), "UNC": unc}, m))
        pts = pick_points(grid)
        result[name] = {"balanced": [pts["balanced"][0], pts["balanced"][1]],
                        "a90": [pts["a90"][0], pts["a90"][1]] if pts["a90"] else None,
                        "a95": [pts["a95"][0], pts["a95"][1]] if pts["a95"] else None}
        print(f"\n== {name} ==")
        print("均衡点 :", fmt(pts["balanced"]))
        print("A≥90点 :", fmt(pts["a90"]))
        print("A≥95点 :", fmt(pts["a95"]))
        # 发布 UNC=0.2 附近的逐阈值表
        print("  UNC=0.2 逐阈值: TH | A_mix | A_only | B关闭")
        for p, m in grid:
            if p["UNC"] == 0.2 and p["TH"] in (0.05, 0.10, 0.15, 0.20, 0.23, 0.25,
                                               0.30, 0.35, 0.40, 0.45, 0.50):
                print(f"  {p['TH']:.2f} | {m['A_mix']*100:5.1f}% | "
                      f"{m['A_only']*100:5.1f}% | {(1-m['B_only'])*100:5.1f}%")

    grid7 = []
    for thr in np.arange(0.20, 0.91, 0.05):
        for hyst in (0.1, 0.2):
            for confirm in (1, 2, 3):
                m = v7b_metrics(rows, round(float(thr), 2), hyst, confirm)
                grid7.append(({"thr": round(float(thr), 2), "hyst": hyst,
                               "confirm": confirm}, m))
    pts7 = pick_points(grid7)
    result["v7b"] = {"balanced": [pts7["balanced"][0], pts7["balanced"][1]],
                     "a90": [pts7["a90"][0], pts7["a90"][1]] if pts7["a90"] else None,
                     "a95": [pts7["a95"][0], pts7["a95"][1]] if pts7["a95"] else None}
    print("\n== v7b ==")
    print("均衡点 :", fmt(pts7["balanced"]))
    print("A≥90点 :", fmt(pts7["a90"]))
    print("A≥95点 :", fmt(pts7["a95"]))
    print("  confirm=2 hyst=0.2 逐阈值: thr | A_mix | A_only | B关闭")
    for p, m in grid7:
        if p["hyst"] == 0.2 and p["confirm"] == 2:
            print(f"  {p['thr']:.2f} | {m['A_mix']*100:5.1f}% | "
                  f"{m['A_only']*100:5.1f}% | {(1-m['B_only'])*100:5.1f}%")

    out = Path(args.rows).with_name("sweep_realhuman.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n工作点: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
