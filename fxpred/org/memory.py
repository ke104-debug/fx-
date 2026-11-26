"""相場の記憶 (情報解析部隊): 23年の情勢・大衆心理・過去の大きな出来事を、今の判断の参考に出す。ルールや重みは変えない。

検証 (reports/history.md、提案 P032〜P035):
- 投機筋 (CFTC、円の先物) の持ち高: 3年の中で極端な円売り (ドル円の買いが混んでいる) の時、20日で -300pips 以上の下げが来た割合は
  2016年以降 20% (ふだん 13%)。巻き戻しの危険の目安。向きの予想にはならない
- 4週間の持ち高の変化はその後20日のドル円と弱く同じ向き (作る +0.09 / 確かめ +0.07、偶然の幅の中) → 参考
- 介入: 円買い介入は5回中4回で20日後も下げ (平均 -418pips)。最初の介入 (2022-09) だけは金利差が広がり続けて戻した
- 日銀の引き締め: 織り込み済み (2023-07・2024-03) は円安、驚き (2022-12・2024-07) は大きな円高
- 情勢の似た過去の日からの予想は効かない (2022年以降マイナス) → 『似た時代』は説明として出すだけ
CFTC は火曜の数字を金曜 (米国時間) に公表。週次の見直しで今年の分を取り直す。"""
import io
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CFTC = os.path.join(ROOT, "config", "cftc_jpy.csv")


def update_cftc() -> int:
    """今年の CFTC (旧形式・金融先物) を取り直して、円の先物の行を足す (無料の公開データ)"""
    y = pd.Timestamp.now(tz="UTC").year
    ua = {"User-Agent": "Mozilla/5.0"}

    def get(url, name_col):
        raw = urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=60).read()
        z = zipfile.ZipFile(io.BytesIO(raw))
        d = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
        return d[d[name_col].str.contains("JAPANESE YEN", na=False)]
    d = get(f"https://www.cftc.gov/files/dea/history/deacot{y}.zip", "Market and Exchange Names")
    dc = [c for c in d.columns if "As of Date in Form YYYY-MM-DD" in c][0]
    new = pd.DataFrame({"date": pd.to_datetime(d[dc]), "spec_long": d["Noncommercial Positions-Long (All)"],
                        "spec_short": d["Noncommercial Positions-Short (All)"], "comm_long": d["Commercial Positions-Long (All)"],
                        "comm_short": d["Commercial Positions-Short (All)"], "oi": d["Open Interest (All)"]})
    try:
        x = get(f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{y}.zip", "Market_and_Exchange_Names")
        tff = pd.DataFrame({"date": pd.to_datetime(x["Report_Date_as_YYYY-MM-DD"]), "lev_long": x["Lev_Money_Positions_Long_All"],
                            "lev_short": x["Lev_Money_Positions_Short_All"], "am_long": x["Asset_Mgr_Positions_Long_All"],
                            "am_short": x["Asset_Mgr_Positions_Short_All"]})
        new = new.merge(tff.drop_duplicates("date"), on="date", how="left")
    except Exception as e:
        print("CFTC (金融先物の分類) を取れませんでした:", e)
    old = pd.read_csv(CFTC, parse_dates=["date"])
    added = (~new.date.isin(old.date)).sum()
    out = pd.concat([old[~old.date.isin(new.date)], new]).sort_values("date")
    out.to_csv(CFTC, index=False, date_format="%Y-%m-%d")
    return int(added)


def crowd() -> dict | None:
    try:
        cf = pd.read_csv(CFTC, parse_dates=["date"]).sort_values("date")
    except Exception:
        return None
    spec = (cf.spec_long - cf.spec_short) / cf.oi
    z = (spec - spec.rolling(156, min_periods=52).mean()) / spec.rolling(156, min_periods=52).std()
    pct = (spec.iloc[-156:] < spec.iat[-1]).mean()
    return {"date": cf.date.iat[-1], "spec": float(spec.iat[-1]), "z": float(z.iat[-1]), "pct": float(pct),
            "chg4": float(spec.iat[-1] - spec.iat[-5]) if len(spec) > 5 else 0.0}


def crowd_line() -> str:
    c = crowd()
    if not c:
        return "投機筋の持ち高: 記録なし"
    side = "円買い (ドル円の売り)" if c["spec"] > 0 else "円売り (ドル円の買い)"
    s = (f"投機筋 (CFTC {c['date']:%m/%d}): {side}が建玉の{abs(c['spec']):.0%} / 3年の中で円買いの多さ {c['pct']:.0%}の位置"
         f" / 4週で{'円買い' if c['chg4'] > 0 else '円売り'}が{abs(c['chg4']):.0%}増")
    if c["z"] <= -1.28:
        s += "\n→ 円売りが極端に混んでいる: 過去 (2016年以降) は20日以内に -300pips 以上の巻き戻しが 20% (ふだん13%)"
    elif c["z"] >= 1.28:
        s += "\n→ 円買いが極端に混んでいる (過去はその後の向きに偏りなし)"
    return s


EVENTS = os.path.join(ROOT, "config", "events_history.csv")


def similar_events(cat: str) -> list[str]:
    """過去の同じ種類の出来事 (介入・日銀・リスクオフなど) のその後。cat は config/events_history.csv の分類"""
    try:
        e = pd.read_csv(EVENTS)
    except Exception:
        return []
    g = e[e.cat.str.contains(cat, regex=False)]
    if g.empty:
        return []
    out = [f"過去の{cat} {len(g)}回: 当日 {g.d1.mean():+.0f} / 5日後 {g.d5.mean():+.0f} / 20日後 {g.d20.mean():+.0f}pips"]
    out += [f"  {r.date}: 当日{r.d1:+.0f} 20日{r.d20:+.0f} ({r.note})" for r in g.itertuples() if isinstance(r.note, str) and r.note]
    return out
