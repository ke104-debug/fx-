"""100万円を運用するシステムの総当たり探索 (23年分・年代ごと)。

システム = 使う作戦の組み合わせ + フィルター + 1回のリスク + 1日の損失停止 + 運用時間 (新規は日本時間10時〜翌1時)
- 新規のエントリーは運用時間内だけ。損切り・利確は注文で入れている前提で時間外も約定 (「25時に手仕舞い」の型も比較)
- 同時に持つのは1つまで。レバレッジは最大25倍。往復コスト0.5pips + 約定のずれ0.3pips (手動発注の遅れ分)
- 各年代を100万円から複利で運用し、年代ごとの倍率・年率・最大下落を出す
"""
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import edge, lab, portfolio as pf, regime as rg, scan, shock as sh, strategies as st, ta_study as ta

COST = 0.8   # pips (スプレッド0.5 + 約定のずれ0.3)
ERAS = [("2004-07", 2004, 2007), ("2008-09 リーマン", 2008, 2009), ("2010-12 円高・介入", 2010, 2012),
        ("2013-15 アベノミクス", 2013, 2015), ("2016-19 低変動", 2016, 2019), ("2020-21 コロナ", 2020, 2021),
        ("2022-26 円安・介入", 2022, 2026)]
OUT = os.path.join(os.path.dirname(__file__), "..", "reports")


def load_all() -> pd.DataFrame:
    from . import longdata
    d = pd.concat([longdata.load(), st.load()]).sort_index()
    return rg.prepare(d[~d.index.duplicated(keep="first")])


def trade(h, i, side, stop, tgt, hold, eod_utc=None):
    """次の足の始値で入る。損切り優先。eod_utc: その UTC 時刻で手仕舞い (作戦固有の当日手仕舞い)"""
    O, H, L, C = h.Open.values, h.High.values, h.Low.values, h.Close.values
    j = i + 1
    e = O[j]
    if side * (e - stop) <= 0 or side * (tgt - e) <= 0:
        return None
    end = min(j + hold, len(h)) - 1
    hours = h.index.hour
    for k in range(j, end + 1):
        if (side == 1 and L[k] <= stop) or (side == -1 and H[k] >= stop):
            px = O[k] if (side == 1 and O[k] < stop) or (side == -1 and O[k] > stop) else stop
            return k, px, "損切り"
        if (side == 1 and H[k] >= tgt) or (side == -1 and L[k] <= tgt):
            px = O[k] if (side == 1 and O[k] > tgt) or (side == -1 and O[k] < tgt) else tgt
            return k, px, "利確"
        if eod_utc is not None and k > j and hours[k] == eod_utc:
            return k, C[k], "当日手仕舞い"
    return end, C[end], "期限"


def trend_breakout_retest(h, i, up):
    """トレンド (20日平均と4時間足が同じ向き) で、直近24本の高値を上抜けた後、6本以内に抜けた水準まで戻って
    支えられた (終値が水準の上) ところで買う。下降トレンドは逆。損切りは水準の0.5ATR外、利確は損切り幅の2倍"""
    if i < 40:
        return None
    side = 1 if up else -1
    for back in range(1, 7):
        b = i - back
        lvl = h.High.iloc[b - 24:b].max() if side == 1 else h.Low.iloc[b - 24:b].min()
        broke = h.Close.iat[b] > lvl if side == 1 else h.Close.iat[b] < lvl
        if not broke:
            continue
        retest = (h.Low.iat[i] <= lvl + 0.1 * h.atr.iat[i] and h.Close.iat[i] > lvl) if side == 1 else \
                 (h.High.iat[i] >= lvl - 0.1 * h.atr.iat[i] and h.Close.iat[i] < lvl)
        if retest:
            stop = lvl - side * 0.5 * h.atr.iat[i]
            e = h.Close.iat[i]
            return side, stop, e + side * 2 * abs(e - stop)
        return None
    return None


def signals(h: pd.DataFrame) -> pd.DataFrame:
    """全作戦のシグナルと、その結果 (入り・決済の時刻と価格)、フィルター用の情報"""
    t0 = time.time()
    d = lab.prep(h)
    up = h.trend_up.values
    c4 = h.Close.resample("4h").last().dropna()
    up4 = (c4.ewm(span=20).mean() > c4.ewm(span=50).mean()).shift(1).reindex(h.index, method="ffill").fillna(False).values
    macd = h.Close.ewm(span=12).mean() - h.Close.ewm(span=26).mean()
    mh = (macd - macd.ewm(span=9).mean()).values
    stoch = d.stoch.values
    atr_ratio = (h.atr / h.atr.rolling(480).mean()).values
    events = st.events(h.index)
    raw = []

    def add(i, key, side, stop, tgt, hold, eod=None, aligned=True):
        r = trade(h, i, side, stop, tgt, hold, eod)
        if r is None:
            return
        k, px, why = r
        e = h.Open.iat[i + 1]
        raw.append((i, key, side, e, stop, tgt, k, px, why, aligned))

    for i in range(700, len(h) - 2):
        r = pf.h_tokyo_break(h, i)
        if r: add(i, "H3", r[0], r[1], r[2], 14, eod=20)
        r = st.range_reversal(h, i, both_sides=True)
        if r: add(i, "H4", r[0], r[1], r[2], 48, aligned=(r[0] == 1) == bool(up[i]))
        r = pf.h_event_momo(h, i)
        if r: add(i, "H5", r[0], r[1], r[0] * 2 * abs(h.Close.iat[i] - r[1]) + h.Close.iat[i], 24)
        if up[i] == up4[i]:
            r = trend_breakout_retest(h, i, bool(up[i]))
            if r: add(i, "TB", r[0], r[1], r[2], 48)
    print(f"  毎時の作戦: {len(raw)}件 ({time.time() - t0:.0f}秒)", flush=True)
    for name, params in (("RB", dict(n=48, touches=3, tgt_frac=0.6, stop_atr=0.45, trend_only=False)),
                         ("RB2", dict(n=72, touches=2, tgt_frac=0.6, stop_atr=0.3, trend_only=True))):
        for (i, side, stop, tgt) in range_levels(d, **params):
            add(i, name, side, stop, tgt, 48, aligned=(side == 1) == bool(up[i]))
    print(f"  レンジ反発: 合計{len(raw)}件 ({time.time() - t0:.0f}秒)", flush=True)
    for side in (1, -1):
        t = sh.find(h, side=side)
        for r in t.itertuples():
            i = h.index.get_loc(r.入り) - 1
            add(i, "S", side, r.損切り, r.利確, 72, aligned=bool(up[h.index.get_loc(r.検知)]) == (side == 1))
    print(f"  急変の反動: 合計{len(raw)}件 ({time.time() - t0:.0f}秒)", flush=True)
    a = pd.DataFrame(raw, columns=["i", "key", "side", "entry", "stop", "target", "exit_i", "exit_px", "why", "aligned"])
    a["time"] = h.index[a.i.values]
    a["exit_time"] = h.index[a.exit_i.values]
    a["pips"] = a.side * (a.exit_px - a.entry) * 100 - COST
    a["risk_pips"] = np.maximum((a.entry - a.stop).abs() * 100, 3)
    a["R"] = a.pips / a.risk_pips
    jst = a.time + pd.Timedelta(hours=10)   # 確定時刻 (足の始まり+1時間) を日本時間に
    a["jst_hour"] = jst.dt.hour
    a["weekday"] = jst.dt.weekday
    a["regime"] = [rg.NAMES[x] for x in h.regime.values[a.i.values]]
    a["macd_agree"] = np.sign(mh[a.i.values]) == a.side
    a["stoch_agree"] = np.where(a.side == 1, stoch[a.i.values] < 30, stoch[a.i.values] > 70)
    a["trend_macd"] = a.macd_agree & ((up[a.i.values] == 1) == (a.side == 1))
    a["atr_ratio"] = atr_ratio[a.i.values]
    pos = events.searchsorted(a.time)
    a["h_to_event"] = [((events[p] - t).total_seconds() / 3600) if p < len(events) else 999 for p, t in zip(pos, a.time)]
    a["era"] = pd.cut(a.time.dt.year, [e[1] - 1 for e in ERAS] + [ERAS[-1][2]], labels=[e[0] for e in ERAS])
    a = a.sort_values("time").reset_index(drop=True)
    print(f"  シグナル合計 {len(a)}件 ({time.time() - t0:.0f}秒)", flush=True)
    return a


def range_levels(d, n=48, touches=3, zone=0.15, tgt_frac=0.6, stop_atr=0.45, adx_max=25, trend_only=False):
    """lab.range_bounce と同じ判定で、(足, 向き, 損切り, 利確) を返す"""
    H, L, C, O, A = d.High.values, d.Low.values, d.Close.values, d.Open.values, d.atr.values
    adx, up = d.adx.values, d.up20d.values
    out = []
    for i in range(max(500, n + 1), len(d) - 2):
        if adx[i] >= adx_max:
            continue
        lo_, hi_ = L[i - n:i].min(), H[i - n:i].max()
        w = hi_ - lo_
        if not (4 * A[i] <= w <= 15 * A[i]):
            continue
        for side in (1, -1):
            if trend_only and (side == 1) != bool(up[i]):
                continue
            lvl = lo_ if side == 1 else hi_
            z = zone * w
            near = (L[i] <= lvl + z) if side == 1 else (H[i] >= lvl - z)
            turned = (C[i] > O[i] and C[i] > lvl) if side == 1 else (C[i] < O[i] and C[i] < lvl)
            if not (near and turned):
                continue
            seg = L[i - n:i] if side == 1 else H[i - n:i]
            hits = np.flatnonzero(seg <= lvl + z) if side == 1 else np.flatnonzero(seg >= lvl - z)
            cnt = 1 + int((np.diff(hits) >= 6).sum()) if len(hits) else 0
            if cnt < touches:
                continue
            out.append((i, side, lvl - side * stop_atr * A[i], lvl + side * tgt_frac * w))
            break
    return out


# ---------------------------------------------------------------- 運用のシミュレーション
def simulate(a: pd.DataFrame, risk: float, daily_stop: int | None, window=(10, 25), flatten=False, h=None, init=1_000_000):
    """a: 使うシグナル (時刻順)。新規は日本時間 window の間だけ。同時保有1つ。daily_stop: その日の損切り回数がこれに達したら新規停止"""
    eq, peak, mdd = float(init), float(init), 0.0
    busy_until = None
    day_losses = {}
    n = wins = 0
    lo, hi = window
    for r in a.itertuples():
        hr = r.jst_hour if r.jst_hour >= lo else r.jst_hour + 24
        if not (lo <= hr < hi):
            continue
        if busy_until is not None and r.time < busy_until:
            continue
        day = (r.time + pd.Timedelta(hours=9 - lo)).date()   # 運用日の区切り (日本時間10時)
        if daily_stop and day_losses.get(day, 0) >= daily_stop:
            continue
        R = r.R
        if flatten and h is not None:
            pass
        f = min(risk, 25 * r.risk_pips / 100 / r.entry)   # 25倍を超えない
        eq *= 1 + f * R
        n += 1; wins += R > 0
        if R <= 0:
            day_losses[day] = day_losses.get(day, 0) + 1
        peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
        busy_until = r.exit_time
        if eq <= init * 0.05:
            break
    return eq / init, mdd, n, (wins / n if n else 0)


FILTERS = {
    "流れの方向だけ (レンジ逆張り・急変)": lambda a: a.aligned | ~a.key.isin(["H4", "S", "RB"]),
    "20日平均+MACDが一致": lambda a: a.trend_macd,
    "指標の2時間前は入らない": lambda a: a.h_to_event >= 2,
    "月曜・金曜は入らない": lambda a: ~a.weekday.isin([0, 4]),
    "値動きが小さすぎる時は入らない": lambda a: a.atr_ratio >= 0.7,
}
STRATS = ["H3", "H4", "H5", "S", "RB", "RB2", "TB"]
STRAT_JA = {"H3": "東京レンジのロンドン抜け", "H4": "レンジ逆張り(従来)", "H5": "指標直後の順張り", "S": "急変の反動",
            "RB": "レンジ反発(2日・3回)", "RB2": "レンジ反発(3日・2回・流れ方向)", "TB": "トレンドのブレイク後の押し"}


def _eval(args):
    a_path, combos = args
    a = pd.read_pickle(a_path)
    fmask = {k: f(a).values for k, f in FILTERS.items()}
    era_idx = {e[0]: (a.era == e[0]).values for e in ERAS}
    out = []
    for strats, filt, risk, dstop in combos:
        m = a.key.isin(strats).values
        for f in filt:
            m = m & fmask[f]
        b = a[m]
        row = {"作戦": "+".join(strats), "フィルター": " / ".join(filt) or "なし", "リスク": risk, "1日の損失停止": dstop or "なし"}
        mults, dds, ns = [], [], []
        for e in ERAS:
            x = b[era_idx[e[0]][m]]
            mult, mdd, n, wr = simulate(x, risk, dstop)
            yrs = e[2] - e[1] + 1 if e[0] != ERAS[-1][0] else 4.9
            row[f"{e[0]}"] = round(mult, 2)
            mults.append(mult); dds.append(mdd); ns.append(n)
        full_mult, full_dd, full_n, full_wr = simulate(b, risk, dstop)
        yrs_total = 22.9
        row.update({"最悪の年代の倍率": round(min(mults), 2), "年代の倍率の中央値": round(float(np.median(mults)), 2),
                    "2倍以上の年代": sum(x >= 2 for x in mults), "プラスの年代": sum(x > 1 for x in mults),
                    "最大下落(年代の最悪)": round(max(dds), 2), "23年通しの倍率": round(full_mult, 1),
                    "23年の年率": round(full_mult ** (1 / yrs_total) - 1, 3) if full_mult > 0 else -1,
                    "23年の最大下落": round(full_dd, 2), "取引回数": full_n, "勝率": round(full_wr, 2)})
        out.append(row)
    return out


def search(a: pd.DataFrame, workers=4) -> pd.DataFrame:
    strat_sets = [c for n in range(1, 5) for c in itertools.combinations(STRATS, n)]
    filt_sets = [c for n in range(0, 3) for c in itertools.combinations(list(FILTERS), n)]
    combos = list(itertools.product(strat_sets, filt_sets, [0.01, 0.02, 0.03, 0.05], [None, 2]))
    path = os.path.join(OUT, "_signals.pkl")
    a.to_pickle(path)
    chunks = [combos[k::workers * 8] for k in range(workers * 8)]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(_eval, [(path, c) for c in chunks])):
            rows += r
            if i % 8 == 0:
                print(f"  探索 {len(rows)}/{len(combos)} ({time.time() - t0:.0f}秒)", flush=True)
    return pd.DataFrame(rows)
