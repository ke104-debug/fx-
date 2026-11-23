"""7つの戦略の型を作って23年で比べ、組み合わせで利益の最大化を目指す。
① 低リスク型 ② 損小多利型 ③ 高勝率型 ④ トレンド追従型 ⑤ ブレイクアウト型 ⑥ 相場選別型 ⑦ 複数戦略併用型
入るのは次の足の始値、損切り優先、往復0.8pips、新規は日本時間10時〜翌1時のみ。"""
import numpy as np
import pandas as pd

from . import lab, sysearch as ss

COST = 0.8


def trail_trade(h, i, side, stop, hold=72, be_at=None, trail_atr=None, chan=None, tgt=None):
    """利益を伸ばす決済: be_at (ATRの何倍進んだら損切りを建値へ)、trail_atr (高値からATRの何倍で追いかける)、
    chan (直近 chan 本の安値/高値で追いかける)。tgt があればそこで利確"""
    O, H, L, C, A = h.Open.values, h.High.values, h.Low.values, h.Close.values, h.atr.values
    j = i + 1
    e = O[j]
    if side * (e - stop) <= 0:
        return None
    a0 = A[i]
    best = e
    end = min(j + hold, len(h)) - 1
    for k in range(j, end + 1):
        if (side == 1 and L[k] <= stop) or (side == -1 and H[k] >= stop):
            px = O[k] if (side == 1 and O[k] < stop) or (side == -1 and O[k] > stop) else stop
            return k, px
        if tgt is not None and ((side == 1 and H[k] >= tgt) or (side == -1 and L[k] <= tgt)):
            return k, tgt
        best = max(best, H[k]) if side == 1 else min(best, L[k])
        if be_at and side * (best - e) >= be_at * a0:
            stop = max(stop, e) if side == 1 else min(stop, e)
        if trail_atr:
            ts = best - side * trail_atr * a0
            stop = max(stop, ts) if side == 1 else min(stop, ts)
        if chan and k - chan > j:
            cs = L[k - chan + 1:k + 1].min() if side == 1 else H[k - chan + 1:k + 1].max()
            stop = max(stop, cs) if side == 1 else min(stop, cs)
    return end, C[end]


def _row(h, i, side, e_stop, res, key):
    k, px = res
    e = h.Open.iat[i + 1]
    risk = max(abs(e - e_stop) * 100, 3)
    return dict(time=h.index[i], exit_time=h.index[k], key=key, side=side, entry=e, risk_pips=risk,
                pips=side * (px - e) * 100 - COST, R=(side * (px - e) * 100 - COST) / risk)


def high_winrate(h, d, tp_atr=0.5, sl_atr=1.5, trend=True):
    """③ 高勝率型: 行き過ぎ (ボリンジャー-2σ かつ RSI<30) からの短い戻りを、小さな利確で取る。大きな流れの方向だけ"""
    out = []
    c, A = h.Close.values, h.atr.values
    mid, sd = h.Close.rolling(20).mean().values, h.Close.rolling(20).std().values
    rsi, up = d.rsi.values, h.trend_up.values
    busy = -1
    for i in range(500, len(h) - 2):
        if i <= busy:
            continue
        for side in (1, -1):
            if trend and (side == 1) != bool(up[i]):
                continue
            if side == 1 and not (c[i] < mid[i] - 2 * sd[i] and rsi[i] < 30): continue
            if side == -1 and not (c[i] > mid[i] + 2 * sd[i] and rsi[i] > 70): continue
            stop = c[i] - side * sl_atr * A[i]
            r = trail_trade(h, i, side, stop, hold=12, tgt=c[i] + side * tp_atr * A[i])
            if r:
                out.append(_row(h, i, side, stop, r, "③高勝率")); busy = r[0]
            break
    return pd.DataFrame(out)


def shock_follow_trail(h, z_th=4.0, sl_atr=1.0, be_at=2.0, trail_atr=2.5):
    """② 損小多利型: 急変に付いていく (流れ方向)。損切り1ATR、2ATR進んだら建値へ、その後は2.5ATRで追いかけて利益を伸ばす"""
    r_ = h.Close.diff().values
    sd = pd.Series(r_).rolling(480).std().values
    c, A, up = h.Close.values, h.atr.values, h.trend_up.values
    out, last_day = [], None
    for i in range(500, len(h) - 2):
        m1, m3, m24 = r_[i], c[i] - c[i - 3], c[i] - c[i - 24]
        z = max(abs(m1) / sd[i], abs(m3) / (sd[i] * np.sqrt(3)), abs(m24) / (sd[i] * np.sqrt(24)))
        if not (z >= z_th or abs(m24) >= 2.0):
            continue
        day = h.index[i].date()
        if day == last_day:
            continue
        side = 1 if (m3 if m3 != 0 else m24) > 0 else -1
        if (side == 1) != bool(up[i]):
            continue
        stop = c[i] - side * sl_atr * A[i]
        r = trail_trade(h, i, side, stop, hold=120, be_at=be_at, trail_atr=trail_atr)
        if r:
            out.append(_row(h, i, side, stop, r, "②損小多利(伸ばす)")); last_day = day
    return pd.DataFrame(out)


def trend_follow(h, n=55, exit_n=20, sl_atr=2.0, need4h=True):
    """④ トレンド追従型: 大きな流れ (20日平均) と4時間足がそろった方向に、直近 n 本の高値 (安値) を抜けたら入り、
    直近 exit_n 本の安値 (高値) を割るまで持ち続ける (トレンドが続く限り伸ばす)"""
    H, L, c, A, up = h.High.values, h.Low.values, h.Close.values, h.atr.values, h.trend_up.values
    c4 = h.Close.resample("4h").last().dropna()
    up4 = (c4.ewm(span=20).mean() > c4.ewm(span=50).mean()).shift(1).reindex(h.index, method="ffill").fillna(False).values
    out, busy = [], -1
    for i in range(500, len(h) - 2):
        if i <= busy:
            continue
        side = 1 if up[i] else -1
        if need4h and up4[i] != up[i]:
            continue
        lvl = H[i - n:i].max() if side == 1 else L[i - n:i].min()
        if not ((c[i] > lvl) if side == 1 else (c[i] < lvl)):
            continue
        stop = c[i] - side * sl_atr * A[i]
        r = trail_trade(h, i, side, stop, hold=240, chan=exit_n)
        if r:
            out.append(_row(h, i, side, stop, r, "④トレンド追従")); busy = r[0]
    return pd.DataFrame(out)


def pullback_trail(h, d):
    """④' トレンド追従 (押し目買い・戻り売り): 流れの方向に、ストキャスの行き過ぎから MACD が戻り始めたら入り、追いかけて伸ばす"""
    out, busy = [], -1
    c, A, up, mh, st = h.Close.values, h.atr.values, h.trend_up.values, d.mh.values, d.stoch.values
    for i in range(500, len(h) - 2):
        if i <= busy:
            continue
        side = 1 if up[i] else -1
        if not ((st[i - 6:i + 1].min() < 25) if side == 1 else (st[i - 6:i + 1].max() > 75)):
            continue
        if not (side * mh[i] > 0 and side * mh[i - 1] <= 0):
            continue
        stop = (d.swing_lo.iat[i] - 0.3 * A[i]) if side == 1 else (d.swing_hi.iat[i] + 0.3 * A[i])
        r = trail_trade(h, i, side, stop, hold=120, be_at=1.5, trail_atr=2.5)
        if r:
            out.append(_row(h, i, side, stop, r, "④'押し目・戻り(伸ばす)")); busy = r[0]
    return pd.DataFrame(out)


def breakout_mtf(h, d, n=48, touches=2):
    """⑤ ブレイクアウト型: 何度も触れたレンジを抜けたら順張り。ダマシ対策に、大きな流れ・4時間足が同じ向きの時だけ。
    損切りはレンジの中央、利確はレンジ幅1倍、届いたら建値へ移して残りを追いかける"""
    c4 = h.Close.resample("4h").last().dropna()
    up4 = (c4.ewm(span=20).mean() > c4.ewm(span=50).mean()).shift(1).reindex(h.index, method="ffill").fillna(False).values
    up = h.trend_up.values
    out, busy = [], -1
    for (i, side, stop, tgt) in ss.range_breakout(d, n=n, touches=touches):
        if i <= busy or (side == 1) != bool(up[i]) or (side == 1) != bool(up4[i]):
            continue
        w = abs(tgt - h.Close.iat[i])
        r = trail_trade(h, i, side, stop, hold=96, be_at=w / h.atr.iat[i], trail_atr=2.0)
        if r:
            out.append(_row(h, i, side, stop, r, "⑤ブレイクアウト(時間足そろう)")); busy = r[0]
    return pd.DataFrame(out)


def build_all(h) -> pd.DataFrame:
    d = lab.prep(h)
    a = pd.read_pickle("reports/_signals_all.pkl")
    win = lambda t: t[((t.time + pd.Timedelta(hours=9)).dt.hour >= 10) | ((t.time + pd.Timedelta(hours=9)).dt.hour < 1)]
    parts = []
    # 既存: E (安定), E2 (損小利大), 指標直後
    for name, kw in (("①低リスク(E)", dict(z_th=3.5)), ("②損小多利(E2)", dict(z_th=4, skip_mf=False, stop_atr=1.0, tgt_atr=4.0))):
        t = lab.shock_follow(d, **kw)
        rows = []
        for r in t.itertuples():
            i = h.index.get_loc(r.time)
            rows.append(dict(time=r.time, key=name, side=r.side, entry=h.Open.iat[i + 1], risk_pips=max(r.risk, 3),
                             pips=r.pips - 0.3, R=(r.pips - 0.3) / max(r.risk, 3)))
        x = pd.DataFrame(rows); x["exit_time"] = x.time + pd.Timedelta(hours=24); parts.append(x)
    ev = a[a.key == "H5"][["time", "exit_time", "side", "entry", "risk_pips", "pips", "R"]].assign(key="指標直後の順張り")
    parts.append(ev)
    for fn, args in ((shock_follow_trail, (h,)), (high_winrate, (h, d)), (trend_follow, (h,)), (pullback_trail, (h, d)), (breakout_mtf, (h, d))):
        x = fn(*args)
        print(f"  {x.key.iat[0] if len(x) else fn.__name__}: {len(x)}回", flush=True)
        parts.append(x)
    t = win(pd.concat(parts, ignore_index=True)).sort_values("time").reset_index(drop=True)
    t["year"] = t.time.dt.year
    t["era"] = pd.cut(t.year, [e[1] - 1 for e in ss.ERAS] + [ss.ERAS[-1][2]], labels=[e[0] for e in ss.ERAS])
    t["price"] = h.Close.reindex(t.time).values
    t["cap"] = 25 * t.risk_pips / 100 / t.price
    return t


def simulate(t: pd.DataFrame, risk: float, max_pos: int = 1, init: float = 1.0, start=None, end=None, cap25=True):
    """複数ポジション対応の運用シミュレーション。入る時の資金 (確定済み) に対して risk の損失になる量で入る。
    同時保有は max_pos まで。決済の時に損益を確定。戻り値: (最終資金, 最大下落, 取引回数, 資金の推移)"""
    x = t
    if start is not None:
        x = x[(x.time >= start) & (x.time < end)]
    eq, peak, mdd, n = init, init, 0.0, 0
    open_pos = []   # (exit_time, pnl_amount)
    curve = []
    for r in x.itertuples():
        # 入る前に、決済済みのポジションの損益を確定
        still = []
        for et, pnl in sorted(open_pos):
            if et <= r.time:
                eq += pnl
                peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak if peak > 0 else 1)
            else:
                still.append((et, pnl))
        open_pos = still
        if eq <= init * 0.01:
            break
        if len(open_pos) >= max_pos:
            continue
        f = min(risk, r.cap) if cap25 else risk
        open_pos.append((r.exit_time, eq * f * r.R))
        n += 1
        curve.append((r.time, eq))
    for et, pnl in sorted(open_pos):
        eq += pnl; peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak if peak > 0 else 1)
    return eq / init, mdd, n, curve


def regime_select(t: pd.DataFrame, keys, min_r=0.05, min_share=0.6) -> pd.DataFrame:
    """⑥ 相場選別型: 各年代で、それより前の年代の成績が良かった作戦だけを使う (未来を見ない)"""
    eras = [e[0] for e in ss.ERAS]
    keep = []
    for k, e in enumerate(eras):
        if k == 0:
            continue
        past = t[t.era.isin(eras[:k]) & t.key.isin(keys)]
        by = past.groupby(["key", "era"], observed=True).R.mean().unstack()
        good = [s for s in by.index if past[past.key == s].R.mean() >= min_r and (by.loc[s] > 0).mean() >= min_share]
        keep.append(t[(t.era == e) & t.key.isin(good)])
    return pd.concat(keep).sort_values("time")
