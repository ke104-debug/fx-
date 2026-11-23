"""テクニカル指標・時間帯・ドル全体の強弱から特徴量を作る"""
import numpy as np
import pandas as pd

from .config import HORIZON, PAIRS


def _rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def usd_strength(data: dict[str, pd.DataFrame]) -> pd.Series:
    """各ペアのリターンを「ドル高 = プラス」にそろえて平均したもの"""
    rets = []
    for t, df in data.items():
        r = np.log(df["Close"]).diff()
        rets.append(r if t.startswith("USD") else -r)
    return pd.concat(rets, axis=1).mean(axis=1)


def build(df: pd.DataFrame, ticker: str, usd: pd.Series, horizon: int = HORIZON) -> pd.DataFrame:
    c, h, l = df["Close"], df["High"], df["Low"]
    lr = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for k in (1, 2, 4, 8, 16, 32, 96):
        f[f"ret_{k}"] = np.log(c / c.shift(k))
    for k in (16, 96):
        f[f"vol_{k}"] = lr.rolling(k).std()
    f["vol_ratio"] = f["vol_16"] / f["vol_96"]
    f["rsi_14"] = _rsi(c, 14)
    f["rsi_56"] = _rsi(c, 56)
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    f["macd"] = macd / c
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / c
    for k in (20, 96):
        m, s = c.rolling(k).mean(), c.rolling(k).std()
        f[f"z_{k}"] = (c - m) / s
    f["range"] = (h - l) / c
    f["range_ratio"] = f["range"] / f["range"].rolling(96).mean()
    f["pos_in_day"] = (c - l.rolling(96).min()) / (h.rolling(96).max() - l.rolling(96).min())
    sign = 1 if ticker.startswith("USD") else -1   # ペア方向にそろえたドル強弱
    u = usd.reindex(df.index) * sign
    for k in (1, 4, 16):
        f[f"usd_{k}"] = u.rolling(k).sum()
    f["hour_sin"] = np.sin(2 * np.pi * (df.index.hour + df.index.minute / 60) / 24)
    f["hour_cos"] = np.cos(2 * np.pi * (df.index.hour + df.index.minute / 60) / 24)
    f["dow"] = df.index.dayofweek
    f["pair"] = list(PAIRS).index(ticker)
    # 目的変数: horizon 本後の終値が上がっているか
    f["fwd_ret"] = np.log(c.shift(-horizon) / c)
    f["target"] = (f["fwd_ret"] > 0).astype(float).where(f["fwd_ret"].notna())
    f["close"] = c
    f["ticker"] = ticker
    return f


def build_all(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    usd = usd_strength(data)
    frames = [build(df, t, usd) for t, df in data.items()]
    return pd.concat(frames).sort_index()


NON_FEATURES = {"fwd_ret", "target", "close", "ticker"}


def feature_cols(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if c not in NON_FEATURES]
