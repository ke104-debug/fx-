"""たくさんのルールを総当たりで試す研究用の道具 (1時間足・15分足・5分足に共通)。

公平にするための決まり:
- 判断は足が確定してから、入るのは次の足の始値。同じ足で損切りと利確の両方に触れたら損切り
- 往復のコスト (既定 0.8pips) を必ず引く
- 「作る期間」(既定 2004-2015) の成績だけで良いルールを選び、「確かめる期間」(2016-) で通用したかを見る
- 年代ごとに安定しているかも見る
"""
import numpy as np
import pandas as pd
try:
    from numba import njit
except ImportError:   # 5分監視など numba を入れていない所でも動かす (M018: 入っていないため流れ×タイミングの通知が止まっていた)
    def njit(*a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)

ERAS = [("2004-07", 2004, 2007), ("2008-09", 2008, 2009), ("2010-12", 2010, 2012), ("2013-15", 2013, 2015),
        ("2016-19", 2016, 2019), ("2020-21", 2020, 2021), ("2022-26", 2022, 2026)]
BUILD_END = 2015


# ------------------------------------------------------------------ 売買の再現 (高速)
@njit(cache=True)
def run(O, H, L, C, A, sig, allow, sl, tp, trail, be, hold, cost, opp_exit):
    """sig: 各足の合図 (+1 買い / -1 売り / 0 なし)。allow: 新規に入ってよい足。
    sl/tp/trail/be は ATR の倍数 (0以下なら使わない)。hold は最長の保有本数。
    戻り値: 入った足, 決済した足, 向き, R (損切り幅を1とした損益、コスト込み)"""
    n = len(O)
    ent = np.empty(n, np.int64); ext = np.empty(n, np.int64); sd = np.empty(n, np.int8); R = np.empty(n)
    m = 0
    i = 0
    while i < n - 2:
        s = sig[i]
        if s == 0 or not allow[i] or not (A[i] > 0):
            i += 1
            continue
        e = O[i + 1]
        a = A[i]
        stop = e - s * sl * a
        tgt = e + s * tp * a if tp > 0 else np.nan
        best = e
        j = i + 1
        px = np.nan
        while True:
            if (s == 1 and L[j] <= stop) or (s == -1 and H[j] >= stop):
                px = stop
                if (s == 1 and O[j] < stop) or (s == -1 and O[j] > stop):
                    px = O[j]
                break
            if tp > 0 and ((s == 1 and H[j] >= tgt) or (s == -1 and L[j] <= tgt)):
                px = tgt
                break
            if j - i >= hold or j >= n - 1:
                px = C[j]
                break
            if opp_exit and sig[j] == -s:
                px = C[j]
                break
            if s == 1:
                best = max(best, H[j])
            else:
                best = min(best, L[j])
            if be > 0 and s * (best - e) >= be * a:
                stop = max(stop, e) if s == 1 else min(stop, e)
            if trail > 0:
                ts = best - s * trail * a
                stop = max(stop, ts) if s == 1 else min(stop, ts)
            j += 1
        ent[m] = i; ext[m] = j; sd[m] = s
        R[m] = (s * (px - e) - cost) / (sl * a)
        m += 1
        i = j + 1 if j > i else i + 1
    return ent[:m], ext[:m], sd[:m], R[:m]


# ------------------------------------------------------------------ 足と指標
def bars(h1: pd.DataFrame | None = None, tf: str = "1h", src: pd.DataFrame | None = None) -> pd.DataFrame:
    """OHLC (UTC) から指標をまとめて作る。src を tf にまとめ直す"""
    d = src if tf is None else src.resample(tf, label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    c, hi, lo = d.Close, d.High, d.Low
    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    for n in (5, 10, 20, 50, 100, 200):
        d[f"ema{n}"] = c.ewm(span=n).mean()
    d["sd20"] = c.rolling(20).std()
    d["mid20"] = c.rolling(20).mean()
    delta = c.diff()
    up, dn = delta.clip(lower=0).ewm(alpha=1 / 14).mean(), (-delta.clip(upper=0)).ewm(alpha=1 / 14).mean()
    d["rsi"] = 100 - 100 / (1 + up / dn)
    k = 100 * (c - lo.rolling(14).min()) / (hi.rolling(14).max() - lo.rolling(14).min())
    d["stoch"] = k.rolling(3).mean()
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    d["mh"] = macd - macd.ewm(span=9).mean()
    pdm = (hi.diff()).where((hi.diff() > -lo.diff()) & (hi.diff() > 0), 0.0)
    ndm = (-lo.diff()).where((-lo.diff() > hi.diff()) & (-lo.diff() > 0), 0.0)
    atr14 = tr.ewm(alpha=1 / 14).mean()
    pdi, ndi = 100 * pdm.ewm(alpha=1 / 14).mean() / atr14, 100 * ndm.ewm(alpha=1 / 14).mean() / atr14
    d["adx"] = (100 * (pdi - ndi).abs() / (pdi + ndi)).ewm(alpha=1 / 14).mean()
    d["bbw_rank"] = (d.sd20 / c).rolling(500, min_periods=100).rank(pct=True)
    d["ret1_z"] = delta / delta.rolling(500, min_periods=100).std()
    # 大きな流れ (日足相当): 1時間足480本 ≒ 20日。足の長さに合わせて本数を変える
    # 2026-11-27 修正 (M015): 以前は最初の2本の間隔で決めていて、1分足から作った15分足・5分足では週末の切れ目をまたいで「1日1本」になり、
    # 「20日平均の向き」が実際は「20本の向き」になっていた。いちばん多い間隔 (最頻値) で決める
    per_day = pd.Timedelta("1D") / pd.Series(d.index[1:] - d.index[:-1]).mode().iat[0] if len(d) > 1 else 24
    per_day = int(round(min(per_day, 288)))
    ema_d = c.ewm(span=20 * per_day).mean()
    d["up_d"] = ema_d.diff(per_day) > 0                     # 20日平均が上向き
    ema_4 = c.ewm(span=max(int(20 * per_day / 6), 2)).mean()
    d["up_4h"] = ema_4.diff(max(per_day // 6, 1)) > 0       # 4時間足の20本平均が上向き相当
    d["jst_h"] = (d.index + pd.Timedelta(hours=9)).hour
    d["wd"] = (d.index + pd.Timedelta(hours=9)).weekday
    d["year"] = d.index.year
    d.attrs["per_day"] = per_day
    return d


# ------------------------------------------------------------------ 合図 (順張り・逆張り)
def _x(a, b):
    """a が b を上抜け (+1) / 下抜け (-1)"""
    return np.where((a > b) & (a.shift() <= b.shift()), 1, np.where((a < b) & (a.shift() >= b.shift()), -1, 0))


def signals(d: pd.DataFrame) -> dict:
    """名前 → (種類, 合図の配列)。種類: 順張り / 逆張り / 時間"""
    c, hi, lo = d.Close, d.High, d.Low
    S = {}
    # --- 順張り
    for f, s in ((5, 20), (10, 50), (20, 100), (50, 200)):
        S[f"平均線クロス{f}/{s}"] = ("順張り", _x(d[f"ema{f}"], d[f"ema{s}"]))
    for n in (20, 55, 100):
        hh, ll = hi.rolling(n).max().shift(), lo.rolling(n).min().shift()
        S[f"{n}本の高値安値ブレイク"] = ("順張り", np.where(c > hh, 1, np.where(c < ll, -1, 0)))
    S["ボリンジャー2σ抜け(順張り)"] = ("順張り", np.where(c > d.mid20 + 2 * d.sd20, 1, np.where(c < d.mid20 - 2 * d.sd20, -1, 0)))
    for n in (20, 48):
        hh, ll = hi.rolling(n).max().shift(), lo.rolling(n).min().shift()
        sq = d.bbw_rank.shift() < 0.2
        S[f"もみ合い後のブレイク{n}本"] = ("順張り", np.where(sq & (c > hh), 1, np.where(sq & (c < ll), -1, 0)))
    S["MACDクロス"] = ("順張り", np.sign(d.mh).diff().fillna(0).clip(-1, 1).astype(int).values)
    for z in (2.5, 3.5, 5.0):
        S[f"急な動きに付いていく{z}σ"] = ("順張り", np.where(d.ret1_z >= z, 1, np.where(d.ret1_z <= -z, -1, 0)))
    S["ADX強まり+向き"] = ("順張り", np.where((d.adx > 25) & (d.adx.shift() <= 25), np.where(d.ema20 > d.ema50, 1, -1), 0))
    # --- 逆張り
    for lv in (20, 30):
        S[f"RSI{lv}/{100 - lv}の逆張り"] = ("逆張り", np.where((d.rsi < lv) & (d.rsi.shift() >= lv), 1,
                                                          np.where((d.rsi > 100 - lv) & (d.rsi.shift() <= 100 - lv), -1, 0)))
    for k in (2, 3):
        S[f"ボリンジャー{k}σの逆張り"] = ("逆張り", np.where(c < d.mid20 - k * d.sd20, 1, np.where(c > d.mid20 + k * d.sd20, -1, 0)))
    S["ストキャス逆張り"] = ("逆張り", np.where((d.stoch < 10), 1, np.where(d.stoch > 90, -1, 0)))
    for z in (2.5, 3.5, 5.0):
        S[f"急な動きの戻り{z}σ"] = ("逆張り", np.where(d.ret1_z <= -z, 1, np.where(d.ret1_z >= z, -1, 0)))
    for n in (20, 48):
        hh, ll = hi.rolling(n).max().shift(), lo.rolling(n).min().shift()
        rng = (hh - ll) / d.atr
        # 安値に触れて反発 (足の終値が安値圏から戻った) → 買い
        S[f"レンジの端で反発{n}本"] = ("逆張り", np.where((lo <= ll) & (c > ll + 0.2 * d.atr) & (rng > 4), 1,
                                                    np.where((hi >= hh) & (c < hh - 0.2 * d.atr) & (rng > 4), -1, 0)))
        S[f"ブレイク失敗の逆張り{n}本"] = ("逆張り", np.where((hi.shift() > hh.shift()) & (c < hh.shift()), -1,
                                                      np.where((lo.shift() < ll.shift()) & (c > ll.shift()), 1, 0)))
    S["平均線から離れすぎの逆張り"] = ("逆張り", np.where((c - d.ema50) / d.atr < -4, 1, np.where((c - d.ema50) / d.atr > 4, -1, 0)))
    return {k: (t, np.asarray(v, dtype=np.int8)) for k, (t, v) in S.items()}


FILTERS = {
    "なし": lambda d, s: np.ones(len(d), bool),
    "大きな流れと同じ向き": lambda d, s: (s == 1) == d.up_d.values,
    "大きな流れと逆向き": lambda d, s: (s == 1) != d.up_d.values,
    "4時間と日足がそろう向き": lambda d, s: ((s == 1) == d.up_d.values) & ((s == 1) == d.up_4h.values),
    "ADX<20 (もみ合い)": lambda d, s: d.adx.values < 20,
    "ADX>25 (流れが強い)": lambda d, s: d.adx.values > 25,
    "値動きが大きい時": lambda d, s: d.bbw_rank.values > 0.7,
    "値動きが小さい時": lambda d, s: d.bbw_rank.values < 0.3,
}
EXITS = {
    "損1:利1": dict(sl=1.0, tp=1.0, trail=0, be=0, hold=48),
    "損1:利2": dict(sl=1.0, tp=2.0, trail=0, be=0, hold=96),
    "損1:利3": dict(sl=1.0, tp=3.0, trail=0, be=0, hold=144),
    "損1.5:利1 (高勝率)": dict(sl=1.5, tp=1.0, trail=0, be=0, hold=48),
    "損2:利4": dict(sl=2.0, tp=4.0, trail=0, be=0, hold=144),
    "損1・追いかけ2": dict(sl=1.0, tp=0, trail=2.0, be=1.0, hold=240),
    "損2・追いかけ3": dict(sl=2.0, tp=0, trail=3.0, be=2.0, hold=480),
    "損1・時間12本": dict(sl=1.0, tp=0, trail=0, be=0, hold=12),
}
WINDOWS = {"10時〜翌1時": lambda d: (d.jst_h.values >= 10) | (d.jst_h.values < 1), "いつでも": lambda d: np.ones(len(d), bool)}


def era_of(years):
    out = np.empty(len(years), object)
    for name, a, b in ERAS:
        out[(years >= a) & (years <= b)] = name
    return out


def search(d: pd.DataFrame, tf: str, cost_pips=0.8, windows=("10時〜翌1時",), sig=None) -> tuple[pd.DataFrame, dict]:
    """合図 × 絞り込み × 決済 × 時間帯 を総当たり。戻り値: 成績表, 取引の明細 (キー → DataFrame)"""
    O, H, L, C, A = (d[k].values.astype(np.float64) for k in ("Open", "High", "Low", "Close", "atr"))
    yrs = d.year.values
    sig = sig or signals(d)
    rows, store = [], {}
    cost = cost_pips / 100
    for sname, (kind, s) in sig.items():
        for fname, ff in FILTERS.items():
            fm = ff(d, s)
            for wname in windows:
                allow = fm & WINDOWS[wname](d)
                for ename, ex in EXITS.items():
                    e, x, sd, R = run(O, H, L, C, A, s, allow, ex["sl"], ex["tp"], ex["trail"], ex["be"],
                                      ex["hold"], cost, False)
                    if len(R) < 30:
                        continue
                    y = yrs[e]
                    b = y <= BUILD_END
                    er = pd.Series(R).groupby(era_of(y)).mean()
                    key = f"{tf}|{sname}|{fname}|{ename}|{wname}"
                    nb, nv = b.sum(), (~b).sum()
                    rb = R[b].mean() if nb else np.nan
                    rows.append(dict(key=key, 足=tf, 合図=sname, 種類=kind, 絞り込み=fname, 決済=ename, 時間帯=wname,
                                     回数=len(R), 年間=len(R) / (yrs[-1] - yrs[0] + 1), 勝率=(R > 0).mean(), 平均R=R.mean(),
                                     作る_R=rb, 作る_控えめ=rb - 1.64 * R[b].std() / np.sqrt(max(nb, 1)) if nb > 1 else np.nan,
                                     確かめ_R=R[~b].mean() if nv else np.nan, 確かめ_回数=nv,
                                     プラス年代=int((er > 0).sum()), 年代数=len(er)))
                    store[key] = (e, x, sd, R)
    return pd.DataFrame(rows), store
