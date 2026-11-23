"""20年分 (Dukascopy, 2004〜2024 + Yahoo 2024〜) の総合検証。結果を reports/validate20.md に保存し、要約を返す。

  1. データのそろい具合
  2. 作戦 x 相場状態 x 5年ごとの成績 (ルールを決めた 2024-26 年以外は、決める時に見ていない期間)
  3. 今の監視ルール (scan.RULES) で各期間を運用した場合
  4. AIチーム (担当AI+戦略選択AI) のウォークフォワード: 毎年、前の4年だけで学習してその年を運用
  5. 急変の反動作戦: 急変の基準を「2円固定」と「価格に比例 (150円で2円)」で比較
"""
import os

import numpy as np
import pandas as pd

from . import agents as ag, longdata, regime as rg, scan, shock as sh, strategies as st

OUT = os.path.join(os.path.dirname(__file__), "..", "reports", "validate20.md")
KEY = {"S": "S 急変の反動(流れの方向)", "H3": "H3 東京レンジのロンドン抜け", "H4": "H4 レンジ逆張り+流れフィルタ", "H5": "H5 指標発表直後の順張り"}
PERIODS = [2003, 2008, 2013, 2018, 2023, 2026]
LABELS = ["2004-08", "2009-13", "2014-18", "2019-23", "2024-26"]


def coverage() -> tuple[int, int, list[str]]:
    import glob
    files = glob.glob(os.path.join(longdata.DIR, "USDJPY_*_BID_hour.bi5"))
    have = {os.path.basename(f)[7:14] for f in files}
    want = [f"{y}_{m:02d}" for y in range(2004, 2025) for m in range(1, 13)]
    missing = [w for w in want if w not in have]
    return len(want) - len(missing), len(want), missing


def run() -> str:
    got, total, missing = coverage()
    lines = [f"# 20年検証レポート", "", f"データ: {got}/{total} か月 (Dukascopy 1時間足) + Yahoo 2024-02〜"]
    if missing:
        lines.append(f"欠けている月: {', '.join(missing[:20])}{' ほか' if len(missing) > 20 else ''}")
    d = pd.concat([longdata.load(), st.load()]).sort_index()
    d = d[~d.index.duplicated(keep="first")]
    h = rg.prepare(d)
    a = ag.candidate_trades(h)
    a["期間"] = pd.cut(a.entry.dt.year, PERIODS, labels=LABELS)
    # 2. 作戦 x 状態 x 期間
    g = a.groupby(["作戦", "状態", "期間"], observed=True).R.agg(["mean", "size"])
    tbl = g.apply(lambda r: f"{r['mean']:+.2f} ({int(r['size'])})", axis=1).unstack("期間").fillna("")
    lines += ["", "## 作戦 x 相場状態 x 期間 (1回平均R と回数)", "", tbl.to_markdown()]
    # 3. 今の監視ルール
    plan = {(KEY[k], s): r for (k, s), r in scan.RULES.items()}
    rows = []
    for per, sub in a.groupby("期間", observed=True):
        cv, t = rg.run_switching(sub, plan)
        if len(cv) == 0:
            continue
        cv.index = pd.DatetimeIndex(cv.index)
        yrs = max((sub.entry.max() - sub.entry.min()).days / 365.25, 0.5)
        rows.append({"期間": per, "最終(万円)": round(cv.iloc[-1] / 1e4), "年率": f"{(cv.iloc[-1] / 1e6) ** (1 / yrs) - 1:+.0%}",
                     "最大下落": f"{(1 - cv / cv.cummax()).max():.0%}", "取引": len(t)})
    lines += ["", "## 今の監視ルールで各期間を100万円から運用", "", pd.DataFrame(rows).to_markdown(index=False)]
    # 4. AIチーム
    p = ag.walk_forward(a, first_test_year=2008, train_years=4)
    res = []
    for mode, label in (("ai", "AIチーム"), ("rules", "固定ルール"), ("all", "候補を全部 (各1%)")):
        cv, t = ag.simulate(p, mode)
        if cv.empty:
            res.append({"運用": label, "取引": 0}); continue
        cv.index = pd.DatetimeIndex(cv.index)
        y = cv.resample("YE").last().ffill()
        yr = (y / y.shift(1).fillna(1e6) - 1)
        res.append({"運用": label, "最終(万円)": round(cv.iloc[-1] / 1e4), "最大下落": f"{(1 - cv / cv.cummax()).max():.0%}",
                    "取引": len(t), "勝率": f"{(t.R > 0).mean():.0%}", "プラスの年": f"{(yr > 0).sum()}/{len(yr)}"})
    lines += ["", "## AIチーム (毎年、前の4年だけで学習) 2008年〜", "", pd.DataFrame(res).to_markdown(index=False)]
    sel = p[p.予想R.notna()]
    if len(sel):
        sel = sel.assign(採点=pd.cut(sel.予想R, [-9, 0, 0.05, 0.2, 9], labels=["マイナス", "0〜0.05", "0.05〜0.2", "0.2以上"]))
        cal = sel.groupby(["状態", "採点"], observed=True).R.agg(["mean", "size"]).round(2).unstack("採点")
        lines += ["", "担当AIの採点ごとの実際の平均R (採点が高いほど実際も良ければ、学習できている)", "", cal.to_markdown()]
    # 5. 急変の反動: 2円固定 vs 価格に比例
    up = h.Close.ewm(span=480).mean().diff(24) > 0
    srows = []
    for name, scale in (("2円固定", False), ("価格に比例", True)):
        out = []
        for side in (1, -1):
            for per_lo, per_hi in zip(PERIODS[:-1], PERIODS[1:]):
                sub = h[(h.index.year > per_lo) & (h.index.year <= per_hi)]
                if len(sub) < 1000:
                    continue
                drop = 2.0 * float(sub.Close.median()) / 150 if scale else 2.0
                t = sh.find(sub, drop=drop, side=side)
                if t.empty:
                    continue
                t = t[up.reindex(t.検知).values == (side == 1)]
                out.append(t.assign(期間=f"{per_lo + 1}-{str(per_hi)[2:]}"))
        if out:
            t = pd.concat(out)
            for per, gp in t.groupby("期間"):
                srows.append({"基準": name, "期間": per, "回数": len(gp), "勝率": f"{(gp.R > 0).mean():.0%}", "平均R": round(gp.R.mean(), 2)})
    lines += ["", "## 急変の反動 (流れの方向だけ)", "", pd.DataFrame(srows).to_markdown(index=False)]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    # Discord 用の要約
    summ = [f"データ {got}/{total} か月"]
    summ += [f"・{r['運用']}: 100万→{r.get('最終(万円)', '-')}万 最大下落{r.get('最大下落', '-')} プラスの年{r.get('プラスの年', '-')}" for r in res]
    summ += [f"・今のルール {r['期間']}: {r['年率']} (最大下落{r['最大下落']})" for r in rows]
    return "\n".join(summ)


if __name__ == "__main__":
    import sys
    print(run())
