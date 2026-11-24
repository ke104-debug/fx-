"""data-m1 ブランチの1分足 (Dukascopy) を読み込み、5分足・15分足などにまとめる。
取引のない時間 (出来高0で値が動かない足) は除く。"""
import glob
import os

import numpy as np
import pandas as pd

DIR = os.path.join(os.path.dirname(__file__), "..", "data", "m1")


def load(years=None) -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(os.path.join(DIR, "USDJPY_*.npz"))):
        y = int(os.path.basename(f)[7:11])
        if years and y not in years:
            continue
        z = np.load(f)
        parts.append(pd.DataFrame({"Open": z["o"], "High": z["h"], "Low": z["l"], "Close": z["c"]},
                                  index=pd.to_datetime(z["t"], unit="s", utc=True)))
    d = pd.concat(parts).astype(np.float64)
    d = d[~d.index.duplicated()].sort_index()
    flat = (d.High == d.Low) & (d.Open == d.Close) & (d.Close == d.Close.shift())
    return d[~flat]


def resample(m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    return m1.resample(tf, label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
