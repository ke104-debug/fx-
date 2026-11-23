"""テクニカル指標の検証: 17の指標を、(A) 単独の方向予想 と (B) 作戦シグナルの補助 (向きが一致したら成績が上がるか) で確かめる。
期間ごとに分けて「期間をまたいで安定して効くか」で判断する (1つの期間だけ良いものは偶然とみなす)。
各指標は「買い寄り=+1 / 売り寄り=-1 / どちらでもない=0」を、その足が確定した時点までのデータだけで出す。"""
import itertools
import os

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "..", "reports", "ta_study.md")


def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def indicators(h: pd.DataFrame) -> pd.DataFrame:
    c, hi, lo, o = h.Close, h.High, h.Low, h.Open
    x = pd.DataFrame(index=h.index)
    sgn = lambda b: np.where(b, 1, -1)
    # --- 流れ (順張り)
    x["移動平均20本の上/下"] = sgn(c > c.rolling(20).mean())
    x["移動平均50/200本の並び"] = sgn(c.ewm(span=50).mean() > c.ewm(span=200).mean())
    c4 = c.resample("4h").last().dropna()
    e4 = (c4.ewm(span=20).mean() > c4.ewm(span=50).mean()).shift(1)       # 確定した4時間足だけ使う
    x["4時間足 移動平均20/50"] = sgn(e4.reindex(h.index, method="ffill").fillna(False).astype(bool))
    x["日足相当 20日平均の向き"] = sgn(c.ewm(span=480).mean().diff(24) > 0)
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    x["MACDヒストグラム (1時間)"] = sgn(macd - macd.ewm(span=9).mean() > 0)
    m4 = c4.ewm(span=12).mean() - c4.ewm(span=26).mean()
    x["MACDヒストグラム (4時間)"] = sgn(((m4 - m4.ewm(span=9).mean()) > 0).shift(1).reindex(h.index, method="ffill").fillna(False).astype(bool))
    r = _rsi(c)
    x["RSI 勢い (55超/45未満)"] = np.where(r > 55, 1, np.where(r < 45, -1, 0))
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    z = (c - mid) / sd
    x["ボリンジャー 勢い (±1σ超え)"] = np.where(z > 1, 1, np.where(z < -1, -1, 0))
    don_h, don_l = hi.rolling(20).max().shift(1), lo.rolling(20).min().shift(1)
    x["20本の高値・安値ブレイク"] = np.where(c > don_h, 1, np.where(c < don_l, -1, 0))
    tenkan = (hi.rolling(9).max() + lo.rolling(9).min()) / 2
    kijun = (hi.rolling(26).max() + lo.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    top, bot = np.maximum(span_a, span_b), np.minimum(span_a, span_b)
    x["一目 雲の上/下"] = np.where(c > top, 1, np.where(c < bot, -1, 0))
    x["一目 転換線/基準線"] = sgn(tenkan > kijun)
    day = h.index.normalize()
    dly = h.groupby(day).agg(H=("High", "max"), L=("Low", "min"), C=("Close", "last")).shift(1)
    piv = ((dly.H + dly.L + dly.C) / 3).reindex(day).values
    x["前日のピボットの上/下"] = np.where(np.isnan(piv), 0, sgn(c.values > piv))
    # --- 行き過ぎ (逆張り)
    x["RSI 逆張り (30未満/70超)"] = np.where(r < 30, 1, np.where(r > 70, -1, 0))
    x["ボリンジャー 逆張り (±2σ超え)"] = np.where(z < -2, 1, np.where(z > 2, -1, 0))
    k = 100 * (c - lo.rolling(14).min()) / (hi.rolling(14).max() - lo.rolling(14).min())
    k3 = k.rolling(3).mean()
    x["ストキャスティクス 逆張り"] = np.where(k3 < 20, 1, np.where(k3 > 80, -1, 0))
    # --- ローソク足の形
    body = (c - o).abs(); rng = (hi - lo).replace(0, np.nan)
    bull_eng = (c > o) & (c.shift() < o.shift()) & (c >= o.shift()) & (o <= c.shift())
    bear_eng = (c < o) & (c.shift() > o.shift()) & (c <= o.shift()) & (o >= c.shift())
    x["包み足"] = np.where(bull_eng, 1, np.where(bear_eng, -1, 0))
    lower_w = np.minimum(c, o) - lo; upper_w = hi - np.maximum(c, o)
    x["ピンバー (長いヒゲ)"] = np.where((lower_w > 2 * body) & (lower_w > 0.6 * rng), 1, np.where((upper_w > 2 * body) & (upper_w > 0.6 * rng), -1, 0))
    return x.fillna(0).astype(int)


def periods(index) -> pd.Series:
    """2.8年分なら半年ごと、20年分なら年代ごと"""
    span = (index.max() - index.min()).days / 365
    if span > 6:
        bins = [2003, 2007, 2009, 2012, 2015, 2019, 2021, 2026]
        lab = ["2004-07", "2008-09", "2010-12", "2013-15", "2016-19", "2020-21", "2022-26"]
        return pd.cut(pd.Series(index.year, index=index), bins, labels=lab)
    return pd.Series(index.year.astype(str) + np.where(index.month > 6, "後半", "前半"), index=index)


def standalone(h, x, horizon=4) -> pd.DataFrame:
    """(A) 指標の向きどおりに horizon 時間後に動いたか (指標が0の時は数えない)"""
    fwd = np.sign(h.Close.shift(-horizon) - h.Close)
    per = periods(h.index)
    rows = []
    for col in x.columns:
        s = x[col]
        m = (s != 0) & fwd.notna() & (fwd != 0)
        hit = (s[m] == fwd[m])
        by = hit.groupby(per[m], observed=True).mean()
        rows.append({"指標": col, "正解率": f"{hit.mean():.1%}", "回数": int(m.sum()), "期間ごとの正解率": " ".join(f"{v:.0%}" for v in by),
                     "50%を超えた期間": f"{(by > 0.5).sum()}/{len(by)}"})
    # 多数決: 流れ系の指標の過半数が同じ向き
    trend_cols = [c for c in x.columns if "逆張り" not in c and c not in ("包み足", "ピンバー (長いヒゲ)")]
    vote = np.sign(x[trend_cols].sum(axis=1))
    m = (vote != 0) & fwd.notna() & (fwd != 0)
    hit = vote[m] == fwd[m]; by = hit.groupby(per[m], observed=True).mean()
    rows.append({"指標": "流れ系の指標の多数決", "正解率": f"{hit.mean():.1%}", "回数": int(m.sum()), "期間ごとの正解率": " ".join(f"{v:.0%}" for v in by),
                 "50%を超えた期間": f"{(by > 0.5).sum()}/{len(by)}"})
    strong = x[trend_cols].sum(axis=1)
    m = (strong.abs() >= len(trend_cols) - 1) & fwd.notna() & (fwd != 0)
    hit = np.sign(strong[m]) == fwd[m]; by = hit.groupby(per[m], observed=True).mean()
    rows.append({"指標": "流れ系がほぼ全部そろった時", "正解率": f"{hit.mean():.1%}", "回数": int(m.sum()), "期間ごとの正解率": " ".join(f"{v:.0%}" for v in by),
                 "50%を超えた期間": f"{(by > 0.5).sum()}/{len(by)}"})
    return pd.DataFrame(rows)


def as_filter(a: pd.DataFrame, x: pd.DataFrame, h: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(B) 作戦シグナルの補助: 指標の向きがシグナルと一致した時 / 逆の時の平均 pips を期間ごとに比べる"""
    per = periods(h.index)
    a = a.copy()
    a["period"] = per.reindex(a.time).values
    st_ = x.reindex(a.time).values * a.side.values[:, None]   # +1: 一致 / -1: 逆 / 0: 中立
    agree = pd.DataFrame(st_, columns=x.columns, index=a.index)
    rows = []
    for col in x.columns:
        g = a.assign(ag=agree[col].values)
        y, n = g[g.ag > 0], g[g.ag < 0]
        if len(y) < 30 or len(n) < 30:
            continue
        diff_by = (y.groupby("period", observed=True).pips.mean() - n.groupby("period", observed=True).pips.mean()).dropna()
        rows.append({"指標": col, "一致した時の平均": round(y.pips.mean(), 1), "回数": len(y), "逆の時の平均": round(n.pips.mean(), 1), "回数 ": len(n),
                     "差": round(y.pips.mean() - n.pips.mean(), 1), "一致の方が良かった期間": f"{(diff_by > 0).sum()}/{len(diff_by)}",
                     "_consistency": (diff_by > 0).mean() if len(diff_by) else 0})
    single = pd.DataFrame(rows).sort_values(["_consistency", "差"], ascending=False)
    # 2つの指標が両方とも一致した時
    rows = []
    cols = [r["指標"] for r in single.to_dict("records")]
    for c1, c2 in itertools.combinations(cols, 2):
        m = (agree[c1] > 0) & (agree[c2] > 0)
        g, rest = a[m], a[~m]
        if len(g) < 40:
            continue
        diff_by = (g.groupby("period", observed=True).pips.mean() - rest.groupby("period", observed=True).pips.mean()).dropna()
        x_ = g.pips
        lower = x_.mean() - 1.64 * x_.std(ddof=1) / np.sqrt(len(x_))
        rows.append({"組み合わせ": f"{c1} + {c2}", "両方一致の平均": round(x_.mean(), 1), "回数": len(g), "それ以外の平均": round(rest.pips.mean(), 1),
                     "良かった期間": f"{(diff_by > 0).sum()}/{len(diff_by)}", "控えめな成績": round(lower, 1), "_c": (diff_by > 0).mean()})
    pairs = pd.DataFrame(rows).sort_values(["_c", "控えめな成績"], ascending=False)
    return single.drop(columns="_consistency"), pairs.drop(columns="_c")


def run(h: pd.DataFrame | None = None) -> str:
    from . import research as rs, regime as rg, strategies as st, longdata
    if h is None:
        d = pd.concat([longdata.load(), st.load()]).sort_index()
        h = rg.prepare(d[~d.index.duplicated(keep="first")])
    x = indicators(h)
    a = rs.candidates(h)
    live = a[rs.live_mask(a)]
    sa = standalone(h, x)
    single_all, pairs_all = as_filter(a, x, h)
    single_live, _ = as_filter(live, x, h)
    md = ["# テクニカル指標の検証", "", f"期間: {h.index[0]:%Y-%m-%d}〜{h.index[-1]:%Y-%m-%d} / 期間の区切り: {', '.join(map(str, periods(h.index).dropna().unique()))}", "",
          "## (A) 単独の方向予想 (4時間後に指標の向きへ動いたか)", "", sa.to_markdown(index=False), "",
          "## (B) 作戦シグナルの補助: 全候補シグナルで、指標の向きが一致した時 vs 逆の時", "", single_all.to_markdown(index=False), "",
          "## (B') 今の運用で通知されるシグナルだけで", "", single_live.to_markdown(index=False), "",
          "## (C) 2つの指標が両方一致した時 (上位20)", "", pairs_all.head(20).to_markdown(index=False), ""]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(md) + "\n")
    return "\n".join(md)


if __name__ == "__main__":
    print(run())
