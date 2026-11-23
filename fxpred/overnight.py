"""夜間の探索結果のまとめ: すべての年代で結果を出したシステム、その共通点、選び方の検証 (前半で選び後半で確認)"""
import os

import numpy as np
import pandas as pd

from .sysearch import ERAS, STRAT_JA

OUT = os.path.join(os.path.dirname(__file__), "..", "reports")
ERA_COLS = [e[0] for e in ERAS]
EARLY, LATE = ERA_COLS[:4], ERA_COLS[4:]   # 2004-2015 / 2016-2026


def ja_strats(s: str) -> str:
    return "・".join(STRAT_JA.get(x, x) for x in s.split("+"))


def summarize(r: pd.DataFrame) -> list[str]:
    r = r.copy()
    r["年代の最小倍率"] = r[ERA_COLS].min(axis=1)
    r["年代の倍率の幾何平均"] = np.exp(np.log(r[ERA_COLS].clip(lower=0.01)).mean(axis=1))
    ok = r[(r["最大下落(年代の最悪)"] <= 0.35)]
    md = [f"評価したシステム: {len(r)}通り", ""]
    # 1. すべての年代でプラス、かつ最大下落35%以内
    allpos = ok[ok["プラスの年代"] == len(ERAS)].sort_values("年代の倍率の幾何平均", ascending=False)
    md += [f"## 1. 7つの年代すべてでプラス、かつどの年代も最大下落35%以内: {len(allpos)}通り", ""]
    cols = ["作戦", "フィルター", "リスク", "1日の損失停止"] + ERA_COLS + ["最大下落(年代の最悪)", "23年通しの倍率", "23年の年率", "取引回数", "勝率"]
    if len(allpos):
        show = allpos.head(15)[cols].copy()
        show["作戦"] = show["作戦"].map(ja_strats)
        md += [show.to_markdown(index=False), ""]
    # 2倍以上
    dbl = ok[ok["2倍以上の年代"] == len(ERAS)]
    md += [f"すべての年代で2倍以上 (各年代を100万円から始めて): {len(dbl)}通り", ""]
    best_dbl = ok.sort_values(["2倍以上の年代", "年代の最小倍率"], ascending=False).head(10)
    show = best_dbl[cols + ["2倍以上の年代"]].copy(); show["作戦"] = show["作戦"].map(ja_strats)
    md += ["2倍以上になった年代が多い順 (最大下落35%以内):", "", show.to_markdown(index=False), ""]
    # 2. 共通点
    top = allpos.head(50) if len(allpos) >= 10 else ok.sort_values("年代の最小倍率", ascending=False).head(50)
    com = []
    for s in sorted({x for v in r["作戦"] for x in v.split("+")}):
        share_top = top["作戦"].str.split("+").apply(lambda v: s in v).mean()
        share_all = r["作戦"].str.split("+").apply(lambda v: s in v).mean()
        com.append({"項目": "作戦", "設定": STRAT_JA.get(s, s), "上位での割合": share_top, "全体での割合": share_all})
    fl = sorted({x for v in r["フィルター"] for x in v.split(" / ") if x != "なし"})
    for f in fl:
        com.append({"項目": "フィルター", "設定": f, "上位での割合": top["フィルター"].str.contains(f, regex=False).mean(),
                    "全体での割合": r["フィルター"].str.contains(f, regex=False).mean()})
    for col in ("リスク", "1日の損失停止"):
        for v in r[col].astype(str).unique():
            com.append({"項目": col, "設定": v, "上位での割合": (top[col].astype(str) == v).mean(), "全体での割合": (r[col].astype(str) == v).mean()})
    c = pd.DataFrame(com)
    c["倍率"] = (c["上位での割合"] / c["全体での割合"]).round(2)
    for k in ("上位での割合", "全体での割合"):
        c[k] = (c[k] * 100).round(0).astype(int).astype(str) + "%"
    md += ["## 2. 上位50のシステムの共通点 (倍率が1より大きいほど上位に多い)", "", c.sort_values("倍率", ascending=False).to_markdown(index=False), ""]
    # 3. 選び方の検証
    early_score = np.exp(np.log(r[EARLY].clip(lower=0.01)).mean(axis=1))
    sel = r.assign(_s=early_score, _min=r[EARLY].min(axis=1))
    sel = sel[(sel[EARLY] > 1).all(axis=1)].sort_values(["_min", "_s"], ascending=False).head(20)
    late_geo = np.exp(np.log(sel[LATE].clip(lower=0.01)).mean(axis=1))
    md += ["## 3. 選び方の検証: 2004〜2015年 (4年代) だけで選んだ上位20が、2016〜2026年 (3年代) でどうだったか", "",
           f"後半もすべての年代でプラスだった: {(sel[LATE] > 1).all(axis=1).sum()}/{len(sel)}、後半の倍率の幾何平均の中央値: {late_geo.median():.2f}", ""]
    show = sel[cols].head(10).copy(); show["作戦"] = show["作戦"].map(ja_strats)
    md += [show.to_markdown(index=False), ""]
    return md
