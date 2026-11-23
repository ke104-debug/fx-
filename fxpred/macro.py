"""経済・市場心理の代理指標 (金利・リスク選好・商品・株価) を Yahoo Finance から取得する"""
import numpy as np
import pandas as pd
import yfinance as yf

MACRO = {  # ティッカー -> 意味
    "^TNX": "米10年債利回り", "^IRX": "米3か月債利回り", "^VIX": "VIX(恐怖指数)",
    "^GSPC": "S&P500", "^N225": "日経平均", "GC=F": "金", "CL=F": "原油(WTI)",
    "DX-Y.NYB": "ドル指数",
}
LEVEL = {"^TNX", "^IRX", "^VIX"}  # 水準の差分で見るもの (それ以外は対数リターン)


def fetch(period: str, interval: str) -> pd.DataFrame:
    d = yf.download(list(MACRO), period=period, interval=interval, progress=False,
                    auto_adjust=False)["Close"]
    d.index = pd.to_datetime(d.index, utc=True) if interval != "1d" else pd.to_datetime(d.index).tz_localize(None)
    return d


def features(m: pd.DataFrame, index: pd.Index, windows=(1, 5, 20)) -> pd.DataFrame:
    """FX の時刻 index にそろえ (直前の値で埋める)、変化量の特徴量を作る"""
    m = m.reindex(m.index.union(index)).ffill().reindex(index)
    f = pd.DataFrame(index=index)
    for t in MACRO:
        key = t.strip("^").replace("=F", "").replace("-Y.NYB", "").lower()
        for k in windows:
            f[f"m_{key}_{k}"] = m[t].diff(k) if t in LEVEL else np.log(m[t] / m[t].shift(k))
    f["m_vix_level"] = m["^VIX"]
    f["m_curve"] = m["^TNX"] - m["^IRX"]  # 長短金利差
    return f
