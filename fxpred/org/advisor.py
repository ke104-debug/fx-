"""進言AI: 組織を見渡して、足りない部分・弱っている部分を見つけて進言する (毎週の見直しに載せる)。
- 部門ごとに「判断に使える (重みのある) 専門AI」がいるか
- 影の運用の記録 (journal/org.csv) で、各専門AIの実際の成績 (意見とその後24時間の動き) を採点し、
  重みがあるのに外れ続けている専門AI / 重み0なのに当たり続けている専門AI を挙げる
- 統括AIの結論が、今の仕組み (シグナル) より良いか悪いか
- まだ使えていないデータ"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MISSING = ["経済指標の結果と市場予想の差 (無料の過去データが無い。Forex Factory の結果を毎週ためていく必要)",
           "CFTC の投機筋の持ち高 (週1回・無料。GitHub 経由で取得できる)",
           "日本の短期金利・2年金利 (日次の無料データが少ない)",
           "財務省の介入の記録 (月末に公表)"]


def live_scores(min_rows=48) -> pd.DataFrame:
    """影の運用の記録から、専門AIごとに「意見の向き × その後24時間の値動き」を採点"""
    p = os.path.join(ROOT, "journal", "org.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    d = pd.read_csv(p)
    if len(d) < min_rows:
        return pd.DataFrame()
    d["time_utc"] = pd.to_datetime(d.time_utc, utc=True)
    d = d.set_index("time_utc").sort_index()
    # 24時間後の価格 (記録の間隔が1時間でも30分でも使えるよう、時刻で探す)
    px = d.price
    fut = pd.Series(px.reindex(px.index + pd.Timedelta(hours=24), method="nearest", tolerance=pd.Timedelta(minutes=45)).values,
                    index=px.index)
    r = (fut - px) * 100        # pips
    rows = []
    skip = {"price", "overall", "verdict", "caution", "risk", "conclusion"}
    for c in d.columns:
        if c in skip or d[c].dtype == object:
            continue
        a = d[c]
        k = a.notna() & r.notna() & (a != 0)
        if k.sum() < 20:
            continue
        rows.append(dict(name=c, n=int(k.sum()), hit=float((np.sign(a[k]) == np.sign(r[k])).mean()),
                         pips=float((np.sign(a[k]) * r[k]).mean())))
    a = d.overall
    k = (a.abs() >= 0.25) & r.notna()
    if k.sum() >= 10:
        rows.append(dict(name="統括AI (強さ0.25以上)", n=int(k.sum()), hit=float((np.sign(a[k]) == np.sign(r[k])).mean()),
                         pips=float((np.sign(a[k]) * r[k]).mean())))
    return pd.DataFrame(rows)


def advise() -> list[str]:
    W = json.load(open(os.path.join(ROOT, "config", "org_weights.json"))).get("weights", {})
    from .meeting import DEPTS
    out = []
    by_dept = {}
    for k, v in W.items():
        by_dept.setdefault(DEPTS.get(k.split("|")[0], k.split("|")[0]), []).append((k, v))
    for dept, xs in by_dept.items():
        live = [k for k, v in xs if v > 0]
        out.append(f"{dept}: 判断に使える専門AI {len(live)}/{len(xs)}人" + ("" if live else " → 判断に使える専門AIがいない。新しい材料の検証が必要"))
    s = live_scores()
    if len(s):
        for r in s.itertuples():
            w = W.get(r.name, None)
            if w and w > 0 and r.n >= 40 and r.pips < 0:
                out.append(f"「{r.name.split('|')[-1]}」は重みがあるのに実際の記録で外れ気味 ({r.n}回・平均{r.pips:+.1f}pips)。重みを下げる候補")
            if (w == 0 or w is None) and r.n >= 80 and r.pips > 3 and r.hit > 0.55:
                out.append(f"「{r.name.split('|')[-1]}」は重み0だが実際の記録で当たっている ({r.n}回・的中{r.hit:.0%}・平均{r.pips:+.1f}pips)。23年で再検証する候補")
        top = s[s.name == "統括AI (強さ0.25以上)"]
        if len(top):
            t = top.iloc[0]
            out.append(f"統括AIの実際の成績: {t.n}回・的中{t.hit:.0%}・24時間後 平均{t.pips:+.1f}pips")
    else:
        out.append("影の運用の記録がまだ少ない (2日分以上たまると採点を始める)")
    out.append("まだ使えていないデータ: " + " / ".join(MISSING[:2]))
    return out


if __name__ == "__main__":
    print("\n".join(advise()))
