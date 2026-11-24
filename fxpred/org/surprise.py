"""指標のサプライズ度を測る担当 (情報解析部隊)。
本当は「結果 − 市場予想」で測りたいが、無料の Forex Factory のフィードには結果が入らない。
そこで、発表直後のドル円の反応 (発表から15分・60分の動き、普段の動きの何倍か) を「市場が感じたサプライズ」として記録する。
記録: journal/events.csv (指標名・予想・前回・15分/60分の反応)。たまったら「反応が大きい時に順張りが効くか」を検証する。"""
import os

import numpy as np
import pandas as pd

PATH = os.path.join(os.path.dirname(__file__), "..", "..", "journal", "events.csv")


def record(now=None) -> list[dict]:
    now = now or pd.Timestamp.now(tz="UTC")
    try:
        from .. import calendar as cal
        ev = [e for e in cal.events(impacts=("High",)) if pd.Timedelta(hours=1) <= now - e["time"] <= pd.Timedelta(hours=8)]
    except Exception:
        return []
    if not ev:
        return []
    old = pd.read_csv(PATH) if os.path.exists(PATH) else pd.DataFrame(columns=["time_utc", "title"])
    done = set(zip(old.time_utc.astype(str), old.title.astype(str)))
    ev = [e for e in ev if (f"{e['time']:%Y-%m-%d %H:%M}", e["title"]) not in done]
    if not ev:
        return []
    import yfinance as yf
    m = yf.download("USDJPY=X", period="5d", interval="1m", progress=False, auto_adjust=False)
    m.columns = m.columns.get_level_values(0)
    m.index = pd.to_datetime(m.index, utc=True)
    c = m.Close.dropna()
    sd15 = c.diff(15).abs().rolling(2000, min_periods=200).median()
    rows = []
    for e in ev:
        t = e["time"]
        try:
            p0 = c[:t].iat[-1]
            p15, p60 = c[:t + pd.Timedelta(minutes=15)].iat[-1], c[:t + pd.Timedelta(minutes=60)].iat[-1]
            base = sd15[:t].iat[-1]
        except Exception:
            continue
        rows.append({"time_utc": f"{t:%Y-%m-%d %H:%M}", "title": e["title"], "country": e["country"], "forecast": e["forecast"],
                     "previous": e["previous"], "move15_pips": round((p15 - p0) * 100, 1), "move60_pips": round((p60 - p0) * 100, 1),
                     "surprise_x": round(abs(p15 - p0) / base, 1) if base and not np.isnan(base) else ""})
    if rows:
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        pd.concat([old, pd.DataFrame(rows)]).to_csv(PATH, index=False)
    return rows
