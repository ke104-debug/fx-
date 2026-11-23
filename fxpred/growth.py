"""「5万円を2年で100万円 (20倍)」に近づける方法の検証。
利用者の実績 (2026-06〜11 に 5万→13万、損小多利 + 指標・介入後の順張り) をもとに、
取引の回数を増やす・利益を伸ばす・勝っている時だけ買い増す・レバレッジの使い方を変える、などを23年で試す。

重要: 国内規制の25倍は「同時に持っている全ポジションの合計」にかかる (styles.simulate は1つずつにかけていた)。
ここの simulate は合計で25倍を超えないようにする。
入るのは次の足の始値、同じ足で損切りと利確の両方なら損切り、往復0.8pips、新規は日本時間10時〜翌1時のみ。"""
import numpy as np
import pandas as pd

COST = 0.8


def _path(h, j, side, stop, hold, tgt=None, be_at=None, trail_atr=None, add_at=None, a0=None):
    """j の始値で入り、決済まで追う。add_at (ATRの倍数) 進んだら同じ量を買い増し、その時点で損切りを建値へ。
    戻り値: (決済の足, 決済価格, 買い増し価格 or None)"""
    O, H, L, C = h.Open.values, h.High.values, h.Low.values, h.Close.values
    e = O[j]
    best, add_px = e, None
    end = min(j + hold, len(h)) - 1
    for k in range(j, end + 1):
        if (side == 1 and L[k] <= stop) or (side == -1 and H[k] >= stop):
            px = O[k] if (side == 1 and O[k] < stop) or (side == -1 and O[k] > stop) else stop
            return k, px, add_px
        if tgt is not None and ((side == 1 and H[k] >= tgt) or (side == -1 and L[k] <= tgt)):
            return k, tgt, add_px
        best = max(best, H[k]) if side == 1 else min(best, L[k])
        if add_at and add_px is None and side * (best - e) >= add_at * a0:
            add_px = e + side * add_at * a0            # 指値で買い増し (その足の中で届いた)
            stop = max(stop, e) if side == 1 else min(stop, e)
        if be_at and side * (best - e) >= be_at * a0:
            stop = max(stop, e) if side == 1 else min(stop, e)
        if trail_atr:
            ts = best - side * trail_atr * a0
            stop = max(stop, ts) if side == 1 else min(stop, ts)
    return end, C[end], add_px


def _jst(h):
    j = h.index + pd.Timedelta(hours=9)
    return j.hour.values, j.weekday.values


def signals(h):
    """入る合図を全部作る。戻り値: 行 (i, side, kind, z)"""
    r = h.Close.diff().values
    sd = pd.Series(r).rolling(480).std().values
    c, o = h.Close.values, h.Open.values
    A, up = h.atr.values, h.trend_up.values
    hour, wd = _jst(h)
    ny = h.index.tz_convert("America/New_York")
    nyh, nywd = ny.hour.values, ny.weekday.values
    out = []
    for i in range(500, len(h) - 2):
        if not (hour[i] >= 10 or hour[i] < 1):
            continue
        m1, m3, m24 = r[i], c[i] - c[i - 3], c[i] - c[i - 24]
        z = max(abs(m1) / sd[i], abs(m3) / (sd[i] * np.sqrt(3)), abs(m24) / (sd[i] * np.sqrt(24)))
        side = 1 if (m3 if m3 != 0 else m24) > 0 else -1
        aligned = (side == 1) == bool(up[i])
        # 急変 (z倍)
        if z >= 2.5 or abs(m24) >= 2.0:
            out.append((i, side, "急変", z, aligned, abs(m1), abs(m3)))
        # 米国の発表時刻 (NY 8:30・10:00・14:00 を含む足) の大きな動き = 指標の代わり
        if nywd[i] < 5 and nyh[i] in (8, 10, 14):
            mv = c[i] - o[i]
            if abs(mv) >= 1.0 * A[i]:
                s2 = 1 if mv > 0 else -1
                out.append((i, s2, f"発表NY{nyh[i]}", abs(mv) / A[i], (s2 == 1) == bool(up[i]), abs(mv), abs(m3)))
    return pd.DataFrame(out, columns=["i", "side", "kind", "z", "aligned", "m1", "m3"])


def trades(h, sig, *, kinds=("急変",), z_th=4.0, align="trend", huge=1.5, sl=1.0, tgt=4.0, hold=24,
           be_at=None, trail_atr=None, add_at=None, one_per_day=True, key="x"):
    """合図 → 取引。align: 'trend' (大きな流れと同じ向きだけ) / 'none' / 'trend_or_huge'
    (流れと逆でも、1〜3時間で huge 円以上動いた = 介入・大きな発表のような動きなら付いていく)"""
    c, A = h.Close.values, h.atr.values
    m = sig.kind.isin(kinds) & ((sig.kind != "急変") | (sig.z >= z_th))
    s = sig[m].sort_values(["i", "kind"]).drop_duplicates("i")
    out, last_day, busy = [], None, -1
    for r in s.itertuples():
        i = r.i
        if i <= busy:
            continue
        ok = r.aligned if align == "trend" else True
        if align == "trend_or_huge":
            ok = r.aligned or max(r.m1, r.m3) >= huge
        if not ok:
            continue
        day = h.index[i].date()
        if one_per_day and day == last_day:
            continue
        side = r.side
        stop = c[i] - side * sl * A[i]
        e = h.Open.iat[i + 1]
        if side * (e - stop) <= 0:
            continue
        k, px, add_px = _path(h, i + 1, side, stop, hold, None if tgt is None else c[i] + side * tgt * A[i],
                              be_at, trail_atr, add_at, A[i])
        rp = max(abs(e - stop) * 100, 3)
        pips = side * (px - e) * 100 - COST
        R = pips / rp
        units = 1
        if add_px is not None:
            R += (side * (px - add_px) * 100 - COST) / rp
            units = 2
        out.append(dict(time=h.index[i], exit_time=h.index[k], key=key, kind=r.kind, side=side, risk_pips=rp,
                        R=R, units=units, price=c[i]))
        last_day, busy = day, k
    return pd.DataFrame(out)


def simulate(t, risk=None, lev=None, max_pos=1, init=5.0, start=None, end=None, cap=25.0):
    """確定済み資金に対して、1回 risk (損切りで失う割合) または lev (レバレッジ倍率) で入る。
    同時に持つ全ポジションの合計レバレッジが cap 以下になるように量を減らす (買い増しの分も数える)。
    戻り値: (最終資金, 最大下落, 取引回数)"""
    x = t if start is None else t[(t.time >= start) & (t.time < end)]
    eq = peak = init
    mdd, n, opens = 0.0, 0, []   # (exit_time, pnl, lev_used)
    for r in x.itertuples():
        still = []
        for et, pnl, lv in sorted(opens):
            if et <= r.time:
                eq += pnl; peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
            else:
                still.append((et, pnl, lv))
        opens = still
        if eq <= init * 0.02:
            return eq, 1.0, n
        if len(opens) >= max_pos:
            continue
        room = cap - sum(lv for _, _, lv in opens)
        lev_per_risk = r.price / (r.risk_pips / 100)          # 1%リスク = 何%の建玉か の逆数
        L = lev if lev is not None else risk * lev_per_risk   # 1単位分のレバレッジ
        L = min(L, room / r.units)
        if L <= 0.5:
            continue
        f = L / lev_per_risk                                  # 1単位の損切りで失う割合
        opens.append((r.exit_time, eq * f * r.R, L * r.units))
        n += 1
    for et, pnl, _ in sorted(opens):
        eq += pnl; peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
    return eq, mdd, n


def windows(t, years=2, **kw):
    starts = pd.date_range("2004-01-01", f"{2026 - years}-11-01", freq="MS", tz="UTC")
    res = np.array([simulate(t, start=s, end=s + pd.DateOffset(years=years), **kw)[:2] for s in starts])
    return starts, res[:, 0], res[:, 1]
