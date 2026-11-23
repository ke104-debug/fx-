"""Yahoo Finance から15分足を取得し、data/ に蓄積する (60日より長い履歴を作るため)"""
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from .config import DATA_DIR, INTERVAL, PAIRS, PERIOD


def _closed_bars(df: pd.DataFrame) -> pd.DataFrame:
    # 最後の足がまだ確定していなければ除く
    now = datetime.now(timezone.utc)
    return df[df.index + timedelta(minutes=15) <= now]


def fetch() -> dict[str, pd.DataFrame]:
    raw = yf.download(list(PAIRS), period=PERIOD, interval=INTERVAL,
                      group_by="ticker", progress=False, auto_adjust=False)
    out = {}
    os.makedirs(DATA_DIR, exist_ok=True)
    for t in PAIRS:
        df = raw[t][["Open", "High", "Low", "Close"]].dropna()
        df.index = pd.to_datetime(df.index, utc=True)
        df = _closed_bars(df)
        path = os.path.join(DATA_DIR, t.replace("=X", "") + ".csv")
        if os.path.exists(path):
            old = pd.read_csv(path, index_col=0, parse_dates=True)
            old.index = pd.to_datetime(old.index, utc=True)
            df = pd.concat([old, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_csv(path)
        out[t] = df
    return out
