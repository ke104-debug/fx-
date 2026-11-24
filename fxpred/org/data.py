"""専門AIが使う日ごとのデータ (他市場・金利・リスク選好)。

時刻のそろえ方 (未来の情報を混ぜないため):
- 判断は毎日 日本時間10時。その時点で分かっているのは「前の日の米国市場の終わり (日本時間 早朝)」までの値
- Yahoo の株・商品・指数 (為替ではない) の日足は米国/日本の取引日の日付なので、米国の日付 D は日本の D+1 の10時に使える
  (日経は日本の日付 D の終わり = 日本の D+1 の10時に使える)
- FRED の金利は公表が翌営業日の夕方なので、さらに1日遅らせる (日本の D+2 の10時に使う)
- 日本10年金利 (FRED・月次) は公表の遅れを見て1か月遅らせる
- ドル円は Yahoo の為替日足を使わない (日付が約1日ずれる)。1時間足の日本時間10時の値を使う"""
import io
import os
import urllib.request

import numpy as np
import pandas as pd

YAHOO = {"^GSPC": "米株", "^N225": "日経", "^VIX": "VIX", "GC=F": "金", "CL=F": "原油", "DX-Y.NYB": "ドル指数",
         "^TNX": "米10年(Yahoo)"}
FRED = {"DGS2": "米2年", "DGS10": "米10年", "DFF": "FF金利", "IRLTLT01JPM156N": "日10年"}
CACHE = os.getenv("FX_ORG_CACHE", "state/org_daily.pkl")


def _fred(sid: str) -> pd.Series:
    import time
    for k in range(4):   # FRED はたまに接続を切るので数回やり直す
        try:
            raw = urllib.request.urlopen(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=60).read()
            break
        except Exception:
            if k == 3:
                raise
            time.sleep(5 * (k + 1))
    s = pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True).iloc[:, 0]
    return pd.to_numeric(s, errors="coerce").dropna()


def _wareki(x: str):
    """財務省の CSV の和暦 (S49.9.24 / H1.1.10 / R8.10.30) → 日付"""
    try:
        era, rest = x[0], x[1:]
        y, m, d = (int(v) for v in rest.split("."))
        return pd.Timestamp({"S": 1925, "H": 1988, "R": 2018}[era] + y, m, d)
    except Exception:
        return pd.NaT


def jgb() -> pd.DataFrame:
    """財務省の国債金利 (無料・日次)。過去分 (jgbcm_all.csv) と今月分 (jgbcm.csv)。公表は翌営業日なので1日遅らせる"""
    parts = []
    for f in ("data/jgbcm_all.csv", "jgbcm.csv"):
        try:
            raw = urllib.request.urlopen(f"https://www.mof.go.jp/jgbs/reference/interest_rate/{f}", timeout=60).read()
            d = pd.read_csv(io.BytesIO(raw), encoding="shift_jis", skiprows=1)
            d.index = d.iloc[:, 0].astype(str).map(_wareki)
            parts.append(d[["2年", "10年"]].apply(pd.to_numeric, errors="coerce"))
        except Exception:
            pass
    if not parts:
        return pd.DataFrame()
    j = pd.concat(parts)
    j = j[~j.index.isna()]
    j = j[~j.index.duplicated(keep="last")].sort_index()
    return j.rename(columns={"2年": "日2年", "10年": "日10年(財務省)"}).shift(1)


def fetch(start="2003-06-01", max_age_h=6) -> pd.DataFrame:
    """米国・日本の日付ごとの終値 (そろえる前)。state にキャッシュ"""
    import time
    if os.path.exists(CACHE) and time.time() - os.path.getmtime(CACHE) < max_age_h * 3600:
        return pd.read_pickle(CACHE)
    import yfinance as yf
    y = yf.download(list(YAHOO), start=start, progress=False, auto_adjust=False)["Close"].rename(columns=YAHOO)
    y.index = pd.to_datetime(y.index).tz_localize(None)
    cols = {}
    for sid, name in FRED.items():
        try:
            # 公表の遅れ: 日次は1営業日、月次は1か月 (それぞれの系列の中で1つ後ろにずらす)
            cols[name] = _fred(sid).shift(1)
        except Exception:
            pass
    f = pd.DataFrame(cols)
    try:
        f = pd.concat([f, jgb()], axis=1, sort=True)
    except Exception:
        pass
    d = pd.concat([y, f], axis=1, sort=True)
    os.makedirs(os.path.dirname(CACHE) or ".", exist_ok=True)
    d.to_pickle(CACHE)
    return d


def at_jst10(raw: pd.DataFrame, jp_days: pd.DatetimeIndex) -> pd.DataFrame:
    """日本の日付 (その日の10時に判断) ごとに、その時点で分かっている最新の値を並べる。
    米国・FRED の日付 D の値 → 日本の D+1 以降。日経 (日本の日付 D) → 日本の D+1 以降"""
    x = raw.copy()
    x.index = x.index + pd.Timedelta(days=1)
    x = x.sort_index().ffill()
    pos = x.index.searchsorted(jp_days, side="right") - 1
    out = pd.DataFrame(np.where(pos[:, None] >= 0, x.values[np.clip(pos, 0, None)], np.nan),
                       index=jp_days, columns=x.columns)
    return out


def usdjpy_jst10(h: pd.DataFrame) -> pd.Series:
    """1時間足から、日本時間10時の足の始値 (=10時ちょうどの値)"""
    jst = h.index + pd.Timedelta(hours=9)
    s = h.Open[jst.hour == 10]
    s.index = (s.index + pd.Timedelta(hours=9)).normalize().tz_localize(None)
    return s[~s.index.duplicated()]
