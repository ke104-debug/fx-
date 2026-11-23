"""相場の状態 (荒れ / トレンド / 通常) を判定し、状態ごとに作戦を切り替えて運用する。

状態の定義 (結果を見る前に決めたもの):
  荒れ    : 直近24時間の値幅が2円以上、または値動きの大きさ (ATR) が普段 (20日平均) の1.8倍以上
  トレンド: 荒れではなく、20日平均線が5日間で1%以上動いている
  通常    : 上のどちらでもない (レンジになりやすい)
"""
import numpy as np
import pandas as pd

from . import portfolio as pf
from . import shock as sh
from . import strategies as st

NAMES = {"shock": "荒れ", "trend": "トレンド", "normal": "通常"}


def label(d: pd.DataFrame) -> pd.Series:
    rng24 = d.High.rolling(24).max() - d.Low.rolling(24).min()
    atr_ratio = d.atr / d.atr.rolling(480).mean()
    ema = d.Close.ewm(span=480).mean()
    slope = ema / ema.shift(120) - 1
    r = np.where((rng24 >= 2.0 * d.Close / 150) | (atr_ratio >= 1.8), "shock",
                 np.where(slope.abs() >= 0.01, "trend", "normal"))
    return pd.Series(r, d.index)


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    h = st.indicators(d)
    h["trend_up"] = h.Close.ewm(span=480).mean().diff(24) > 0
    ev = st.events(h.index); pos = ev.searchsorted(h.index, side="right") - 1
    prev = pd.Series([ev[p] if p >= 0 else pd.NaT for p in pos], index=h.index)
    h["h_since_event"] = (h.index.to_series() - prev).dt.total_seconds() / 3600 - 1
    h["regime"] = label(h)
    return h


def all_trades(h: pd.DataFrame, diff=None) -> pd.DataFrame:
    """全作戦の取引を集め、入った時点の相場状態を付ける"""
    out = []
    for name, rule in pf.HOURLY.items():
        if name.startswith("H6"):
            continue
        _, tr = pf.run_hourly(h, rule, diff=diff)
        if len(tr):
            out.append(tr[["entry", "exit", "side", "R", "dist", "price"]].assign(作戦=name))
    t = sh.trend_aligned(h)
    if len(t):
        out.append(pd.DataFrame({"entry": t.入り, "exit": t.入り, "side": 0, "R": t.R, "dist": t.dist, "price": t.price,
                                 "作戦": "S 急変の反動(流れの方向)"}))
    a = pd.concat(out)
    reg = h.regime
    # 判定は入る直前の確定足で行う (未来を見ない)
    a["状態"] = [NAMES[reg.asof(e - pd.Timedelta(hours=1))] for e in a.entry]
    return a.sort_values("entry").reset_index(drop=True)


def by_regime(a: pd.DataFrame) -> pd.DataFrame:
    g = a.groupby(["作戦", "状態"]).R.agg(回数="size", 平均R="mean").round(2).unstack("状態")
    return g


def run_switching(a: pd.DataFrame, plan: dict, init=1_000_000) -> tuple[pd.Series, pd.DataFrame]:
    """plan = {(作戦, 状態): 1回のリスク}。該当する取引だけを複利で積み上げる"""
    t = a[[(s, r) in plan for s, r in zip(a.作戦, a.状態)]].copy()
    t["risk"] = [plan[(s, r)] for s, r in zip(t.作戦, t.状態)]
    eq, path = init, []
    for R, dist, px, f, ts in zip(t.R, t.dist, t.price, t.risk, t.entry):
        eq *= 1 + min(f, pf.MAX_LEV * dist / px) * R
        path.append((ts, eq))
    return pd.Series(dict(path)), t
