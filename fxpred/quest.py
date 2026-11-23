"""5万円を大きく増やすシステムの探索 (目標100万円 = 20倍。3年 → 達成したら期間を短くして再挑戦)。

公平にするための決まり:
- 設定は「作る期間」2004-2015 だけで選び、「確かめる期間」2016-2026 の結果で合否を判断する
- 達成の判定: その期間の長さの運用を毎月始めたとして、確かめる期間の始点の **半分以上** で20倍に届くこと
  (あわせて「届いた割合」「中央値」「元本割れ」「ほぼ全損」を出す)
- 国内規制: 同時に持つ全ポジションの合計で25倍まで。往復0.8pips。スワップ (金利差) も計算に入れる
- 入るのは次の足の始値、同じ足で損切りと利確の両方なら損切り
"""
import numpy as np
import pandas as pd

from . import growth as g

COST = 0.8
# 年ごとのおおよその政策金利 (%)。米国 FF 金利の年平均と日銀。スワップは金利差 ±0.5% (業者の取り分) と仮定
FED = {2004: 1.35, 2005: 3.22, 2006: 4.97, 2007: 5.02, 2008: 1.92, 2009: .16, 2010: .18, 2011: .10, 2012: .14,
       2013: .11, 2014: .09, 2015: .13, 2016: .40, 2017: 1.00, 2018: 1.83, 2019: 2.16, 2020: .38, 2021: .08,
       2022: 1.68, 2023: 5.02, 2024: 5.14, 2025: 4.20, 2026: 3.70}
BOJ = {**{y: 0.0 for y in range(2004, 2006)}, 2006: .2, 2007: .45, 2008: .4, **{y: .1 for y in range(2009, 2016)},
       **{y: -.1 for y in range(2016, 2024)}, 2024: .1, 2025: .4, 2026: .75}
BUILD_END = pd.Timestamp("2016-01-01", tz="UTC")


def swap_pips(t0, t1, side, price):
    """持っていた日数分のスワップ (pips)。買い = 金利差 -0.5%、売り = -(金利差 +0.5%)"""
    days = max((t1 - t0).total_seconds() / 86400, 0)
    if days < 0.2:
        return 0.0
    diff = FED[t0.year] - BOJ[t0.year]
    rate = (diff - 0.5) if side == 1 else -(diff + 0.5)
    return price * rate / 100 * days / 365 * 100


def _row(h, i, k, side, stop, px, key, add=None):
    e = h.Open.iat[i + 1]
    rp = max(abs(e - stop) * 100, 3)
    pips = side * (px - e) * 100 - COST + swap_pips(h.index[i + 1], h.index[k], side, e)
    return dict(time=h.index[i], exit_time=h.index[k], key=key, side=side, risk_pips=rp, R=pips / rp,
                units=1, price=h.Close.iat[i])


# ------------------------------------------------------------------ 作戦の候補
def fam_shock(h, sig, z=4.0, exit="伸ばす", align="trend"):
    EX = {"E2": dict(sl=1, tgt=4, hold=24), "伸ばす": dict(sl=1, tgt=None, hold=120, be_at=2, trail_atr=2.5),
          "大きく": dict(sl=1, tgt=6, hold=48), "長く伸ばす": dict(sl=1, tgt=None, hold=480, be_at=2, trail_atr=4)}
    t = g.trades(h, sig, z_th=z, align=align, key=f"急変z{z}{exit}", **EX[exit])
    t["R"] = t.R + [swap_pips(a, b, s, p) / rp for a, b, s, p, rp in zip(t.time, t.exit_time, t.side, t.price, t.risk_pips)]
    return t


def fam_event(h):
    a = pd.read_pickle("reports/_signals_all.pkl")
    x = a[a.key == "H5"][["time", "exit_time", "side", "risk_pips", "R"]].copy()
    x["units"], x["price"], x["key"] = 1, h.Close.reindex(x.time).values, "指標直後"
    jh = (x.time + pd.Timedelta(hours=9)).dt.hour
    return x[(jh >= 10) | (jh < 1)]


def gotobi_days(idx):
    """五十日 (5・10・15・20・25・月末)。土日なら前の平日"""
    days = pd.DatetimeIndex(sorted(set((idx + pd.Timedelta(hours=9)).normalize().tz_localize(None))))
    biz = days[days.weekday < 5]
    out = set()
    for m, grp in pd.Series(biz, index=biz).groupby([biz.year, biz.month]):
        for d in (5, 10, 15, 20, 25, 31):
            cand = grp[grp.dt.day <= d]
            if len(cand):
                out.add(cand.iloc[-1])
    return out


def fam_gotobi(h, mode="朝の買い", sl_atr=1.0):
    """五十日の仲値 (9:55) に向けた円売り (ドル買い) の需要。
    朝の買い: 日本時間7時に買い、10時に決済 (運用時間外 → 自動売買向け)
    仲値後の売り: 10時に売り、15時に決済 (運用時間内)"""
    gd = gotobi_days(h.index)
    jst = h.index + pd.Timedelta(hours=9)
    day, hr = jst.normalize().tz_localize(None), jst.hour
    A = h.atr.values
    out = []
    if mode == "朝の買い":
        ent, ext, side = 6, 9, 1      # 6時台の足の終値 = 7時に入る、9時台の足の終値 (10時) で決済
    else:
        ent, ext, side = 9, 14, -1
    pos = {}
    for i in range(500, len(h) - 2):
        if day[i] not in gd:
            continue
        if hr[i] == ent:
            pos[day[i]] = i
        elif hr[i] == ext and day[i] in pos:
            i0 = pos.pop(day[i])
            e = h.Open.iat[i0 + 1]
            stop = e - side * sl_atr * A[i0]
            seg = slice(i0 + 1, i + 1)
            hit = (h.Low.values[seg] <= stop).any() if side == 1 else (h.High.values[seg] >= stop).any()
            px = stop if hit else h.Close.iat[i]
            out.append(_row(h, i0, i, side, stop, px, f"五十日{mode}"))
    return pd.DataFrame(out)


def daily(h):
    """日本時間10時の時点の日足 (前日10時〜当日10時)。各日の判断は10時台の足の終値で行い、次の足で入る"""
    jst_h = (h.index + pd.Timedelta(hours=9)).hour
    pts = np.where(jst_h == 10)[0]
    c = h.Close.values
    hi = pd.Series(h.High.values).rolling(24).max().values
    lo = pd.Series(h.Low.values).rolling(24).min().values
    d = pd.DataFrame({"i": pts, "c": c[pts], "hi": hi[pts], "lo": lo[pts]}, index=h.index[pts])
    tr = np.maximum(d.hi - d.lo, (d.hi - d.c.shift()).abs())
    d["atr"] = tr.rolling(20).mean()
    return d


def fam_trend(h, dd, kind="ema", fast=10, slow=50, n=20, trail=3.0, sl=2.0, only=None):
    """日足の流れに乗り続ける (スワップも受け取る/払う)。
    kind=ema: 速い平均と遅い平均の向き、kind=donchian: n日の高値/安値を抜けたら。
    損切りは sl×日足ATR、その後は高値 (安値) から trail×日足ATR で追いかける。only=1 なら買いだけ"""
    c = dd.c.values
    if kind == "ema":
        sig = np.sign(dd.c.ewm(span=fast).mean() - dd.c.ewm(span=slow).mean()).values
    else:
        hh, ll = dd.hi.rolling(n).max().shift(1).values, dd.lo.rolling(n).min().shift(1).values
        sig = np.where(c > hh, 1, np.where(c < ll, -1, 0))
    H, L, O, C = h.High.values, h.Low.values, h.Open.values, h.Close.values
    dpos = np.full(len(h), -1)
    dpos[dd.i.values.astype(int)] = np.arange(len(dd))
    out, k_busy = [], -1
    for j in range(60, len(dd) - 1):
        i = int(dd.i.iat[j])
        if i <= k_busy or sig[j] == 0 or np.isnan(dd.atr.iat[j]):
            continue
        side = int(sig[j])
        if only and side != only:
            continue
        a = dd.atr.iat[j]
        e = O[i + 1]
        stop = e - side * sl * a
        best, k = e, i + 1
        exit_px = None
        while k < len(h) - 1:
            if (side == 1 and L[k] <= stop) or (side == -1 and H[k] >= stop):
                exit_px = O[k] if side * (O[k] - stop) < 0 else stop
                break
            best = max(best, H[k]) if side == 1 else min(best, L[k])
            stop = max(stop, best - trail * a) if side == 1 else min(stop, best + trail * a)
            # 流れが反転したら (日本時間10時の判断で) 決済
            jj = dpos[k]
            if kind == "ema" and jj >= 0 and sig[jj] == -side:
                exit_px = C[k]
                break
            k += 1
        if exit_px is None:
            exit_px = C[k]
        r = _row(h, i, k, side, e - side * sl * a, exit_px, f"流れ{kind}{fast if kind == 'ema' else n}")
        out.append(r)
        k_busy = k
    return pd.DataFrame(out)


# ------------------------------------------------------------------ 運用シミュレーション
def simulate(t, lev, init=5.0, start=None, end=None, cap=25.0):
    """t の行ごとに lev[key] 倍で入る (合計 cap 倍まで)。戻り値 (最終資金, 最大下落)"""
    x = t if start is None else t[(t.time >= start) & (t.time < end)]
    eq = peak = init
    mdd, opens = 0.0, []
    for r in x.itertuples():
        if opens:
            still = []
            for et, pnl, lv in opens:
                if et <= r.time:
                    eq += pnl; peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
                else:
                    still.append((et, pnl, lv))
            opens = still
        if eq <= init * 0.02:
            return eq, 1.0
        room = cap - sum(o[2] for o in opens)
        L = min(lev.get(r.key, 0), room)
        if L <= 0.5:
            continue
        f = L * (r.risk_pips / 100) / r.price
        opens.append((r.exit_time, eq * f * r.R, L))
    for et, pnl, _ in opens:
        eq += pnl; peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
    return eq, mdd


def windows(t, lev, years, first="2004-01-01", last=None):
    end_last = pd.Timestamp("2026-11-23") - pd.DateOffset(months=int(round(years * 12)))
    starts = pd.date_range(first, end_last, freq="MS", tz="UTC")
    span = pd.DateOffset(months=int(round(years * 12)))
    t = t.sort_values("time")
    res = np.array([simulate(t, lev, start=s, end=s + span) for s in starts])
    return starts, res[:, 0], res[:, 1]


def score(t, lev, years, init=5.0):
    s, m, dd = windows(t, lev, years)
    b, v = s < BUILD_END, s >= BUILD_END
    def part(mask):
        x = m[mask]
        return dict(n=int(mask.sum()), hit=float((x >= 100).mean()), x10=float((x >= 50).mean()),
                    med=float(np.median(x)), loss=float((x < 5).mean()), ruin=float((x <= 1).mean()),
                    best=float(x.max()))
    return part(b), part(v), (s, m, dd)
