"""日足の長期検証 (2004年〜): テクニカルのみ vs テクニカル+マクロ を年ごとのウォークフォワードで比較

注意: Yahoo の為替日足は日付が約1日ずれている (行Dの終値 ≒ 実際のD-1終値。ドル指数との相関で確認)。
そのため (1) 為替は終値だけを使い高値/安値は使わない (2) マクロ指標は1日ずらして時刻をそろえる。
ずらさないと未来の情報が混入し、正解率が72%などと見せかけの高精度になる。"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import roc_auc_score

from . import features as ft
from . import macro
from .config import PAIRS
from .model import PARAMS
import lightgbm as lgb


def load():
    raw = yf.download(list(PAIRS), period="max", interval="1d", group_by="ticker",
                      progress=False, auto_adjust=False)
    data = {}
    for t in PAIRS:
        df = raw[t][["Open", "High", "Low", "Close"]].dropna()
        df = df[(df.index >= "2004-01-01") & (df["High"] > df["Low"])]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        data[t] = df
    return data, macro.fetch("max", "1d")


def build(data, m, horizon: int, macro_lag: int):
    usd = ft.usd_strength(data)
    frames = []
    for t, df in data.items():
        f = ft.build(df, t, usd, horizon=horizon)
        f = f.drop(columns=["hour_sin", "hour_cos", "range", "range_ratio", "pos_in_day"])
        mf = macro.features(m, df.index).shift(macro_lag)
        flip = [c for c in mf.columns if c not in ("m_vix_level", "m_curve")]
        if not t.startswith("USD"):  # EUR/USD 等はドル高=下落なので向きをそろえる
            mf[flip] = -mf[flip]
        frames.append(pd.concat([f, mf], axis=1))
    return pd.concat(frames).sort_index()


def walk_forward(f, cols, first_test_year=2012):
    f = f.dropna(subset=["target"])
    out = []
    for y in range(first_test_year, f.index.max().year + 1):
        tr, te = f[f.index.year < y], f[f.index.year == y].copy()
        tr = tr[tr.index < pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=10)]
        m = lgb.LGBMClassifier(**PARAMS).fit(tr[cols], tr["target"].astype(int))
        te["p_up"] = m.predict_proba(te[cols])[:, 1]
        out.append(te)
    return pd.concat(out)


def score(p, th=0.55):
    y = p["target"].astype(int)
    sig = p[(p.p_up >= th) | (p.p_up <= 1 - th)]
    hit = ((sig.p_up > .5) == sig.target).mean() if len(sig) else float("nan")
    # 毎日 確率>0.5 なら買い/それ以外売り、の単純損益 (pips換算は通貨ごとに違うので%で)
    pnl = np.where(p.p_up > .5, 1, -1) * p.fwd_ret
    yearly = pd.Series(pnl, index=p.index).groupby(p.index.year).sum()
    return dict(acc=((p.p_up > .5) == y).mean(), auc=roc_auc_score(y, p.p_up), n=len(p),
                sig_n=len(sig), sig_hit=hit, ret_pct=pnl.sum() * 100 / len(PAIRS),
                win_years=f"{(yearly > 0).sum()}/{len(yearly)}")


def run():
    data, m = load()
    rows = []
    for horizon in (1, 5):
        f = build(data, m, horizon, macro_lag=1)
        tech = [c for c in ft.feature_cols(f) if not c.startswith("m_")]
        macro_only = [c for c in ft.feature_cols(f) if c.startswith("m_") or c in ("pair", "dow")]
        sets = {"テクニカルのみ": tech, "マクロのみ": macro_only, "テクニカル+マクロ": ft.feature_cols(f)}
        for name, cols in sets.items():
            r = score(walk_forward(f, cols))
            rows.append({"予想": f"{horizon}日後", "特徴量": name, **r})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df


if __name__ == "__main__":
    run()
