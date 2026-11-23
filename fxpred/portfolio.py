"""複数の作戦で USD/JPY を運用した想定のシミュレーション (勝率ではなく、増えた資金で評価する)

共通ルール:
- 初期資金 100万円。日本の個人向け規制に合わせ、レバレッジは最大25倍
- スワップ (日米金利差) と取引コストを含める。パラメータは一般的な値で固定し、過去に合わせ込まない
- 日足作戦: 2004年〜。値動きの大きさに合わせて持ち高を調整 (年率リスク10%目標)。終値だけ使う
  (Yahoo の為替日足は高値/安値が当てにならないため)
- 1時間足作戦: 2024年2月〜。1回の損失を資金の1%に抑える取引量。シグナルは足の確定後、次の足で入る
"""
import numpy as np
import pandas as pd
import yfinance as yf

from . import strategies as st

INIT = 1_000_000
MAX_LEV = 25
PIP = 0.01

# 日銀の政策金利 (概算。2025年12月以降の 0.75% は仮定)
BOJ = [("2004-01-01", 0.0), ("2006-07-14", 0.25), ("2007-02-21", 0.5), ("2008-10-31", 0.3), ("2008-12-19", 0.1),
       ("2010-10-05", 0.05), ("2016-02-16", -0.1), ("2024-03-19", 0.05), ("2024-07-31", 0.25),
       ("2025-01-24", 0.5), ("2025-12-19", 0.75)]


def rate_diff(index) -> pd.Series:
    """米3か月金利 - 日銀政策金利 (%)。業者のスワップは金利差より不利なので、受取は8割・支払は12割で計算"""
    irx = yf.download("^IRX", period="max", interval="1d", progress=False, auto_adjust=False)["Close"].squeeze()
    irx.index = pd.to_datetime(irx.index).tz_localize(None)
    boj = pd.Series({pd.Timestamp(d): r for d, r in BOJ}).sort_index()
    idx = pd.DatetimeIndex(index).tz_localize(None) if getattr(index, "tz", None) else pd.DatetimeIndex(index)
    us = irx.reindex(irx.index.union(idx)).ffill().reindex(idx)
    jp = boj.reindex(boj.index.union(idx)).ffill().reindex(idx).fillna(0)
    return pd.Series((us - jp).values, index=index)


def swap_yen(units, price, diff_pct, days):
    """units>0 は買い。1日あたり金利差 x 取引額。受取は8割、支払は12割"""
    rate = diff_pct / 100 * np.where(units * diff_pct > 0, 0.8, 1.2)
    return units * price * rate * days / 365


# ======================= 日足の作戦 (目標ポジション -1〜+1 を返す)

def daily_signals(c: pd.Series, diff: pd.Series) -> dict[str, pd.Series]:
    ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
    hi55, lo55 = c.rolling(55).max().shift(1), c.rolling(55).min().shift(1)
    hi20, lo20 = c.rolling(20).max().shift(1), c.rolling(20).min().shift(1)
    s = {}
    s["D1 移動平均50/200 (順張り)"] = np.sign(ma50 - ma200)
    pos, p = [], 0                                  # 55日高値更新で買い/安値更新で売り、20日の逆側で手仕舞い
    for x, a, b, e, f in zip(c, hi55, lo55, hi20, lo20):
        if p == 0 and x > a: p = 1
        elif p == 0 and x < b: p = -1
        elif p == 1 and x < f: p = 0
        elif p == -1 and x > e: p = 0
        pos.append(p)
    s["D2 ブレイクアウト55/20日"] = pd.Series(pos, c.index, dtype=float)
    mom = np.sign(c / c.shift(252) - 1)            # 12か月モメンタム、月1回だけ見直す
    s["D3 12か月モメンタム"] = mom.where(c.index.to_series().dt.month.diff().fillna(0) != 0).ffill()
    s["D4 金利差+上昇トレンドの時だけ買い"] = ((diff > 0.5) & (c > ma200)).astype(float)
    r = c.diff()
    rsi2 = 100 - 100 / (1 + r.clip(lower=0).ewm(alpha=.5).mean() / (-r.clip(upper=0)).ewm(alpha=.5).mean())
    s["D5 RSI(2)逆張り (短期の行き過ぎ)"] = pd.Series(np.where(rsi2 < 10, 1, np.where(rsi2 > 90, -1, 0)), c.index).where(lambda x: x != 0).ffill(limit=3).fillna(0)
    s["基準: ただ買って持つ (1倍)"] = pd.Series(1.0, c.index)
    return {k: v.fillna(0) for k, v in s.items()}


def run_daily(c: pd.Series, sig: pd.Series, diff: pd.Series, vol_target=0.10, cost_pips=1.0, fixed_lev=None):
    ret = c.pct_change()
    vol = ret.rolling(60).std() * np.sqrt(252)
    eq, out, units = INIT, [], 0.0
    prev_px = c.iloc[0]
    for t, px in c.items():
        days = 1
        eq += units * (px - prev_px) + swap_yen(units, px, diff.get(t, 0), days)
        lev = fixed_lev if fixed_lev else min(vol_target / vol.get(t, np.nan), 5) if vol.get(t, 0) > 0 else 0
        target_units = sig.get(t, 0) * lev * eq / px      # ドル建ての数量
        if abs(target_units - units) * px > 0.05 * eq or (sig.get(t, 0) == 0 and units != 0):
            eq -= abs(target_units - units) * cost_pips * PIP
            units = target_units
        out.append(eq); prev_px = px
    return pd.Series(out, c.index)


# ======================= 1時間足の作戦 (足 i で (方向, 損切り, 利確 or None, 追いかけ決済ルール) を返す)

def h_breakout(d, i):
    r = d.iloc[i]
    if r.Close > d.High.iloc[i - 120:i].max(): return 1, r.Close - 2 * r.atr, None, "low48"
    if r.Close < d.Low.iloc[i - 120:i].min(): return -1, r.Close + 2 * r.atr, None, "high48"


def h_pullback(d, i):
    r = d.iloc[i]
    if r.ema50 > r.ema200 and r.Close > r.ema200 and r.rsi < 35: return 1, r.Close - 1.5 * r.atr, None, "chandelier"
    if r.ema50 < r.ema200 and r.Close < r.ema200 and r.rsi > 65: return -1, r.Close + 1.5 * r.atr, None, "chandelier"


def session_break(d, i, rng_h=(0, 6), trade_h=(7, 11), wmin=15, wmax=60, rr=2, close_h=20):
    """あるセッションの値幅を、次のセッションで抜けた方向へ。反対側で損切り、rr倍で利確、close_h時(UTC)に手仕舞い"""
    t = d.index[i]
    if not (trade_h[0] <= t.hour <= trade_h[1]): return None
    day = d[(d.index >= t.normalize() + pd.Timedelta(hours=rng_h[0])) & (d.index < t.normalize() + pd.Timedelta(hours=rng_h[1]))]
    if len(day) < (rng_h[1] - rng_h[0]) - 1: return None
    hi, lo = day.High.max(), day.Low.min()
    w = (hi - lo) / PIP
    if not (wmin <= w <= wmax): return None
    r, p = d.iloc[i], d.iloc[i - 1]
    if r.Close > hi and p.Close <= hi: return 1, lo, r.Close + rr * (r.Close - lo), f"eod{close_h}"
    if r.Close < lo and p.Close >= lo: return -1, hi, r.Close - rr * (hi - r.Close), f"eod{close_h}"


def h_london_ny(d, i):
    """ロンドン時間 (7-12時UTC) の値幅を、NY 時間の始まり (12-15時UTC) に抜けた方向へ"""
    return session_break(d, i, rng_h=(7, 12), trade_h=(12, 15), wmin=20, wmax=80, rr=2, close_h=21)


def h_tokyo_break(d, i):
    """東京時間 (0-6時UTC) の値幅を、ロンドン時間 (7-11時UTC) に抜けた方向へ。反対側で損切り、2倍で利確"""
    t = d.index[i]
    if not (7 <= t.hour <= 11): return None
    if "tk_hi" in d:   # 事前計算済み (高速版)
        hi, lo, cnt = d.tk_hi.iat[i], d.tk_lo.iat[i], d.tk_n.iat[i]
        if not cnt >= 5: return None
    else:
        day = d[(d.index >= t.normalize()) & (d.index < t.normalize() + pd.Timedelta(hours=6))]
        if len(day) < 5: return None
        hi, lo = day.High.max(), day.Low.min()
    w = (hi - lo) / PIP
    if not (15 <= w <= 60): return None
    r, p = d.iloc[i], d.iloc[i - 1]
    if r.Close > hi and p.Close <= hi: return 1, lo, r.Close + 2 * (r.Close - lo), "eod"
    if r.Close < lo and p.Close >= lo: return -1, hi, r.Close - 2 * (hi - r.Close), "eod"


def h_range_trend(d, i):
    s = st.range_reversal(d, i, both_sides=True)
    if s and ((s[0] == 1) != bool(d.trend_up.iat[i])): return None
    return (s[0], s[1], s[2], None) if s else None


def h_event_momo(d, i):
    """FOMC・米雇用統計の発表直後1時間の方向に付いていく。1日で手仕舞い"""
    r = d.iloc[i]
    if not (-1.5 < r.h_since_event <= 0.5): return None
    move = (r.Close - r.Open) / r.atr
    if move > 1: return 1, r.Close - 1.5 * r.atr, None, "24h"
    if move < -1: return -1, r.Close + 1.5 * r.atr, None, "24h"


HOURLY = {"H1 ブレイクアウト(5日高安値)": h_breakout, "H2 押し目買い/戻り売り": h_pullback,
          "H3 東京レンジのロンドン抜け": h_tokyo_break, "H4 レンジ逆張り+流れフィルタ": h_range_trend,
          "H5 指標発表直後の順張り": h_event_momo, "H6 ロンドンレンジのNY抜け": h_london_ny}


def run_hourly(d, rule, risk=0.01, cost_pips=0.5, diff=None, max_hold=240):
    O, H, L, C, ATR = d.Open.values, d.High.values, d.Low.values, d.Close.values, d.atr.values
    eq, curve, trades, i, n = INIT, [], [], 220, len(d)
    curve_idx = []
    while i < n - 2:
        sig = rule(d, i)
        if not sig:
            curve.append(eq); curve_idx.append(d.index[i]); i += 1; continue
        side, stop, tgt, trail = sig
        j = i + 1; e = O[j]
        dist = abs(e - stop)
        if dist <= 0 or (side == 1 and stop >= e) or (side == -1 and stop <= e):
            i += 1; continue
        units = min(eq * risk / dist, MAX_LEV * eq / e)
        px, why = None, ""
        for k in range(j, min(j + max_hold, n)):
            if side == 1:
                if O[k] <= stop: px, why = O[k], "損切り"
                elif L[k] <= stop: px, why = stop, "損切り"
                elif tgt and H[k] >= tgt: px, why = tgt, "利確"
            else:
                if O[k] >= stop: px, why = O[k], "損切り"
                elif H[k] >= stop: px, why = stop, "損切り"
                elif tgt and L[k] <= tgt: px, why = tgt, "利確"
            if px is not None: break
            # 追いかけ決済 (利益を伸ばす)
            if trail == "low48" and k > j + 1:
                stop = max(stop, L[k - 48:k].min()) if side == 1 else stop
            if trail == "high48" and k > j + 1:
                stop = min(stop, H[k - 48:k].max())
            if trail == "chandelier":
                stop = max(stop, C[k] - 3 * ATR[k]) if side == 1 else min(stop, C[k] + 3 * ATR[k])
            if trail == "eod" and d.index[k].hour >= 20: px, why = C[k], "当日手仕舞い"; break
            if trail and trail.startswith("eod") and trail != "eod" and d.index[k].hour >= int(trail[3:]): px, why = C[k], "当日手仕舞い"; break
            if trail == "24h" and k - j >= 24: px, why = C[k], "1日経過"; break
        if px is None: px, why = C[k], "時間切れ"
        hours = k - j + 1
        sw = swap_yen(side * units, e, diff.iat[j] if diff is not None else 0, hours / 24) if diff is not None else 0
        pnl = side * units * (px - e) - units * cost_pips * PIP + sw
        eq += pnl
        trades.append(dict(entry=d.index[j], exit=d.index[k], side=side, pnl=pnl, why=why, r=pnl / (eq - pnl) / risk,
                           R=(side * (px - e) - cost_pips * PIP) / dist, dist=dist, price=e))
        curve.extend([eq] * (k - i)); curve_idx.extend(d.index[i:k]); i = k
    return pd.Series(curve, curve_idx), pd.DataFrame(trades)


def summarize(curve: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    curve = curve[~curve.index.duplicated()]
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    final = curve.iloc[-1]
    dd = (1 - curve / curve.cummax()).max()
    daily = curve.resample("D").last().dropna().pct_change().dropna()
    out = {"最終資金(万円)": round(final / 1e4, 1), "増減": f"{final / INIT - 1:+.0%}",
           "年率": f"{(final / INIT) ** (1 / years) - 1:+.1%}" if final > 0 else "破綻",
           "最大下落": f"{dd:.0%}", "シャープ": round(daily.mean() / daily.std() * np.sqrt(252), 2) if daily.std() > 0 else 0}
    if trades is not None and len(trades):
        w, l = trades[trades.pnl > 0], trades[trades.pnl <= 0]
        out.update({"回数": len(trades), "勝率": f"{len(w) / len(trades):.0%}",
                    "損益比": round(w.pnl.mean() / -l.pnl.mean(), 2) if len(w) and len(l) else None})
    return out


def yearly(curve: pd.Series) -> pd.Series:
    y = curve.resample("YE").last()
    return (y / y.shift(1).fillna(INIT) - 1).rename(lambda t: t.year)


def run():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    # ---- 日足 2004〜
    raw = yf.download("USDJPY=X", period="max", interval="1d", progress=False, auto_adjust=False)["Close"].squeeze().dropna()
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    c = raw[raw.index >= "2004-01-01"]
    diff = rate_diff(c.index)
    rows, curves = [], {}
    for name, sig in daily_signals(c, diff).items():
        cv = run_daily(c, sig.shift(1).fillna(0), diff, fixed_lev=1 if name.startswith("基準") else None)
        curves[name] = cv
        rows.append({"作戦": name, **summarize(cv)})
    combo = sum(run_daily(c, s.shift(1).fillna(0), diff, vol_target=0.10 / 2) - INIT
                for k, s in daily_signals(c, diff).items() if k.startswith(("D1", "D2", "D4"))) / 1 + INIT
    rows.append({"作戦": "組み合わせ D1+D2+D4 (資金を分けて同時運用)", **summarize(combo)})
    print(f"■ 日足 {c.index[0]:%Y-%m-%d}〜{c.index[-1]:%Y-%m-%d} (約{(c.index[-1]-c.index[0]).days/365.25:.0f}年) / 年率リスク10%目標 / スワップ込み")
    print(pd.DataFrame(rows).to_string(index=False))
    yr = pd.DataFrame({k[:2]: yearly(v) for k, v in curves.items()} | {"組合せ": yearly(combo)})
    print("\n年ごとの増減 (%):"); print((yr * 100).round(0).astype("Int64").T.to_string())

    # ---- 1時間足 2024〜
    h = st.indicators(st.load())
    h["trend_up"] = h.Close.ewm(span=480).mean().diff(24) > 0
    ev = st.events(h.index)
    pos = ev.searchsorted(h.index, side="right") - 1
    prev = pd.Series([ev[p] if p >= 0 else pd.NaT for p in pos], index=h.index)
    h["h_since_event"] = (h.index.to_series() - prev).dt.total_seconds() / 3600 - 1  # イベントを含む足の確定時に0付近
    hdiff = rate_diff(h.index)
    rows, hc = [], {}
    for name, rule in HOURLY.items():
        cv, tr = run_hourly(h, rule, diff=hdiff)
        hc[name] = cv
        rows.append({"作戦": name, **summarize(cv, tr)})
    bh = INIT + (h.Close / h.Close.iloc[0] - 1) * INIT + (swap_yen(INIT / h.Close.iloc[0], h.Close, hdiff, 1 / 24)).cumsum()
    rows.append({"作戦": "基準: ただ買って持つ (1倍)", **summarize(bh)})
    print(f"\n■ 1時間足 {h.index[0]:%Y-%m-%d}〜{h.index[-1]:%Y-%m-%d} / 1回の損失=資金の1% / スワップ込み")
    print(pd.DataFrame(rows).to_string(index=False))
    return curves, hc


if __name__ == "__main__":
    run()
