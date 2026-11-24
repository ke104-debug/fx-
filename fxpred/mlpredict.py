"""「今の相場からこの後どう動くか」を機械学習で予想し、確率が高い時だけ入る検証。
毎年1月に「前年末までのデータ」だけで学び直し、その年を予想する (未来を見ない)。
予想の対象: この後 hold 本の間に「+k ATR に先に届く」か「-k ATR に先に届く」か (買い・売りそれぞれ)。"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import explore as ex


def features(d: pd.DataFrame) -> pd.DataFrame:
    c, A = d.Close, d.atr
    f = pd.DataFrame(index=d.index)
    for n in (1, 3, 6, 12, 24, 72, 120):
        f[f"ret{n}"] = (c - c.shift(n)) / A
    for n in (5, 20, 50, 200):
        f[f"dist_ema{n}"] = (c - d[f"ema{n}"]) / A
    f["slope20"] = d.ema20.diff(5) / A
    f["slope200"] = d.ema200.diff(24) / A
    for n in (24, 120):
        hh, ll = d.High.rolling(n).max(), d.Low.rolling(n).min()
        f[f"pos{n}"] = (c - ll) / (hh - ll)
        f[f"width{n}"] = (hh - ll) / A
    f["rsi"], f["stoch"], f["adx"] = d.rsi, d.stoch, d.adx
    f["mh"] = d.mh / A
    f["bbw_rank"] = d.bbw_rank
    f["atr_ratio"] = A / A.rolling(480, min_periods=50).mean()
    f["z1"] = d.ret1_z
    f["up_d"], f["up_4h"] = d.up_d.astype(int), d.up_4h.astype(int)
    f["hour"], f["wd"] = d.jst_h, d.wd
    f["body"] = (c - d.Open) / A
    f["wick_up"] = (d.High - np.maximum(c, d.Open)) / A
    f["wick_dn"] = (np.minimum(c, d.Open) - d.Low) / A
    return f


def labels(d: pd.DataFrame, k=1.0, hold=24):
    """+k ATR に先に届けば 1 (買いが正解)、-k ATR が先なら 0、どちらも届かなければ NaN。次の足の始値から数える"""
    O, H, L, A = d.Open.values, d.High.values, d.Low.values, d.atr.values
    n = len(d)
    y = np.full(n, np.nan)
    for i in range(n - hold - 1):
        e = O[i + 1]
        up, dn = e + k * A[i], e - k * A[i]
        for j in range(i + 1, i + 1 + hold):
            hit_dn, hit_up = L[j] <= dn, H[j] >= up
            if hit_dn and hit_up:
                y[i] = 0; break       # 同じ足で両方なら損切り側
            if hit_up:
                y[i] = 1; break
            if hit_dn:
                y[i] = 0; break
    return y


def walk_forward(d: pd.DataFrame, k=1.0, hold=24, first_test=2010, step=3):
    """年ごとに学び直して、その年の各足の「買いが正解の確率」を出す"""
    X = features(d)
    y = labels(d, k, hold)
    p = pd.Series(np.nan, index=d.index)
    yrs = d.index.year
    for yr in range(first_test, yrs.max() + 1):
        # 学習データの最後の hold 本は答えが学習期間の外に出るので除く
        tr = (yrs < yr) & ~np.isnan(y)
        tr[np.flatnonzero(yrs < yr)[-hold:] if (yrs < yr).any() else []] = False
        te = yrs == yr
        idx = np.flatnonzero(tr)[::step]
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        m.fit(X.iloc[idx], y[idx])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p, y
