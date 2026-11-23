"""トレンド相場・レンジ相場で利益を上げる作戦の研究室。
作戦の候補をいくつも作り、パラメータを少しずつ変えて全期間に流し、「期間をまたいで安定しているか」で評価する。
入るのはシグナルの次の足の始値、同じ足で損切りと利確の両方に触れたら損切り扱い、往復コスト0.5pips。"""
import itertools
import os

import numpy as np
import pandas as pd

from . import research as rs, ta_study as ta

OUT = os.path.join(os.path.dirname(__file__), "..", "reports", "lab.md")


def prep(h: pd.DataFrame) -> pd.DataFrame:
    d = h.copy()
    c, hi, lo = d.Close, d.High, d.Low
    d["ema20"], d["ema50"] = c.ewm(span=20).mean(), c.ewm(span=50).mean()
    c4 = c.resample("4h").last().dropna()
    d["up4"] = (c4.ewm(span=20).mean() > c4.ewm(span=50).mean()).shift(1).reindex(d.index, method="ffill").fillna(False).astype(bool)
    d["up20d"] = c.ewm(span=480).mean().diff(24) > 0
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    d["mh"] = macd - macd.ewm(span=9).mean()
    k = 100 * (c - lo.rolling(14).min()) / (hi.rolling(14).max() - lo.rolling(14).min())
    d["stoch"] = k.rolling(3).mean()
    d["rsi"] = ta._rsi(c)
    d["swing_lo"], d["swing_hi"] = lo.rolling(12).min(), hi.rolling(12).max()
    return d


def sim(d, i, side, stop, tgt, hold=48):
    pips, res, _ = rs.outcome(d, i + 1, side, stop, tgt, hold)
    return pips, abs(d.Open.iat[i + 1] - stop) * 100


# ---------------------------------------------------------------- トレンド: 押し目買い・戻り売り
def trend_pullback(d, osc="stoch", osc_lv=30, need_macd=True, rr=2.0, stop_buf=0.3, both_tf=True):
    """大きな流れ (20日平均の向き、4時間足もそろう) の方向に、1時間足が20本平均まで押した (戻した) 後、
    行き過ぎ (ストキャス or RSI) から MACD が流れの方向に戻り始めたところで入る。損切りは直近12本の安値 (高値) の少し外"""
    out = []
    A = d.atr.values
    for i in range(500, len(d) - 2):
        up, up4 = d.up20d.iat[i], d.up4.iat[i]
        if both_tf and up != up4:
            continue
        side = 1 if up else -1
        touched = d.Low.iloc[i - 3:i + 1].min() <= d.ema20.iat[i] if side == 1 else d.High.iloc[i - 3:i + 1].max() >= d.ema20.iat[i]
        if not touched:
            continue
        o = d[osc].iloc[i - 6:i + 1]
        stretched = (o.min() <= osc_lv) if side == 1 else (o.max() >= 100 - osc_lv)
        if not stretched:
            continue
        if need_macd and not (side * d.mh.iat[i] > 0 and side * d.mh.iat[i - 1] <= 0):
            continue
        stop = d.swing_lo.iat[i] - stop_buf * A[i] if side == 1 else d.swing_hi.iat[i] + stop_buf * A[i]
        e = d.Close.iat[i]
        if side * (e - stop) <= 0.05:
            continue
        tgt = e + side * rr * abs(e - stop)
        pips, risk = sim(d, i, side, stop, tgt)
        out.append((d.index[i], side, pips, risk))
    return pd.DataFrame(out, columns=["time", "side", "pips", "risk"])


# ---------------------------------------------------------------- レンジ: 下値で買い・上値で売り
def range_bounce(d, n=72, touches=2, zone=0.15, osc_lv=25, need_osc=True, adx_max=25, tgt_frac=0.8, stop_atr=0.5, trend_only=True):
    """直近 n 本の高値・安値を上値・下値とし、それぞれ touches 回以上触れて反発しているレンジだけ使う。
    下値ゾーンで陽線 (上値ゾーンで陰線) の反発を確認し、ストキャスの売られすぎ (買われすぎ) も条件にできる"""
    out = []
    H, L, C, O, A = d.High.values, d.Low.values, d.Close.values, d.Open.values, d.atr.values
    adx, st_ = d.adx.values, d.stoch.values
    up = d.up20d.values
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
            if need_osc and not ((st_[i - 3:i + 1].min() <= osc_lv) if side == 1 else (st_[i - 3:i + 1].max() >= 100 - osc_lv)):
                continue
            stop = lvl - side * stop_atr * A[i]
            tgt = lvl + side * tgt_frac * w
            if side * (C[i] - stop) <= 0.05 or side * (tgt - C[i]) <= 0.05:
                continue
            pips, risk = sim(d, i, side, stop, tgt)
            out.append((d.index[i], side, pips, risk))
            break
    return pd.DataFrame(out, columns=["time", "side", "pips", "risk"])


GRIDS = {
    "トレンド: 押し目買い・戻り売り": (trend_pullback, dict(osc=["stoch", "rsi"], osc_lv=[25, 35], need_macd=[True, False], rr=[1.5, 2.0, 3.0],
                                                        stop_buf=[0.2, 0.5], both_tf=[True, False])),
    "レンジ: 下値で買い・上値で売り": (range_bounce, dict(n=[48, 72, 120], touches=[1, 2, 3], osc_lv=[20, 30], need_osc=[True, False],
                                                     tgt_frac=[0.6, 0.8], stop_atr=[0.3, 0.6], trend_only=[True, False])),
}


def evaluate(t: pd.DataFrame, per: pd.Series) -> dict:
    if len(t) < 30:
        return {"回数": len(t)}
    t = t.assign(p=per.reindex(t.time).values)
    by = t.groupby("p", observed=True).pips.agg(["mean", "size"])
    by = by[by["size"] >= 5]
    lower = [(g.pips.mean() - 1.64 * g.pips.std(ddof=1) / np.sqrt(len(g))) for _, g in t.groupby("p", observed=True) if len(g) >= 5]
    eq = t.pips.cumsum()
    win, loss = t[t.pips > 0].pips.sum(), -t[t.pips <= 0].pips.sum()
    return {"回数": len(t), "勝率": f"{(t.pips > 0).mean():.0%}", "平均pips": round(t.pips.mean(), 1), "合計pips": round(t.pips.sum()),
            "PF": round(win / loss, 2) if loss else None, "最大DD": round((eq.cummax() - eq).max()),
            "プラスの期間": f"{(by['mean'] > 0).sum()}/{len(by)}", "_pos": (by["mean"] > 0).mean() if len(by) else 0,
            "控えめな成績": round(float(np.median(lower)), 1) if lower else None,
            "期間ごとの平均": " ".join(f"{v:+.0f}" for v in by["mean"])}


def run(h: pd.DataFrame | None = None) -> str:
    from . import longdata, regime as rg, strategies as st
    if h is None:
        dd = pd.concat([longdata.load(), st.load()]).sort_index()
        h = rg.prepare(dd[~dd.index.duplicated(keep="first")])
    d = prep(h)
    per = ta.periods(d.index)
    md = ["# トレンド・レンジの作戦の研究", "", f"期間: {d.index[0]:%Y-%m-%d}〜{d.index[-1]:%Y-%m-%d} / 区切り: {', '.join(map(str, per.dropna().unique()))}", ""]
    best = {}
    for name, (fn, grid) in GRIDS.items():
        rows = []
        for vals in itertools.product(*grid.values()):
            p = dict(zip(grid, vals))
            t = fn(d, **p)
            r = evaluate(t, per)
            if r["回数"] >= 30:
                rows.append({**{k: v for k, v in p.items()}, **r})
        res = pd.DataFrame(rows)
        if res.empty:
            md += [f"## {name}", "", "シグナルが少なすぎて評価できず", ""]; continue
        res = res.sort_values(["_pos", "控えめな成績"], ascending=False)
        best[name] = res.iloc[0]
        stable = res[res._pos >= 0.8]
        md += [f"## {name} ({len(res)}通り)", "", f"8割以上の期間でプラスだった設定: {len(stable)}/{len(res)} 通り", "",
               res.head(12).drop(columns=["_pos"]).to_markdown(index=False), ""]
        # 共通点: 安定上位 (8割以上の期間でプラス) に多い設定
        if len(stable) >= 3:
            com = []
            for k in grid:
                for v, cnt in stable[k].astype(str).value_counts().items():
                    base = (res[k].astype(str) == v).mean()
                    com.append({"項目": k, "設定": v, "安定した設定での割合": f"{cnt / len(stable):.0%}", "全体での割合": f"{base:.0%}",
                                "倍率": round(cnt / len(stable) / base, 2)})
            md += ["安定した設定の共通点 (倍率が1より大きいほど安定した設定に多い)", "",
                   pd.DataFrame(com).sort_values("倍率", ascending=False).head(12).to_markdown(index=False), ""]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(md) + "\n")
    return "\n".join(md)


if __name__ == "__main__":
    print(run())


def shock_follow(d, z_th=3.5, yen=2.0, stop_atr=1.5, tgt_atr=3.0, hold=24, aligned=True, skip_mf=True, window=True):
    """23年の研究で全年代プラスだった「急変に付いていく」(reports/overnight.md)。
    1時間・3時間・24時間の動きが普段の z_th 倍以上 (または24時間で yen 円以上) なら、その向きに次の足で入る。
    大きな流れ (20日平均の向き) と同じ向きだけ、月曜・金曜 (日本時間) と運用時間外 (10時〜翌1時以外) は入らない。1日1回まで"""
    from . import research as rs_
    r = d.Close.diff().values
    sd = pd.Series(r).rolling(480).std().values
    c, A = d.Close.values, d.atr.values
    up = d.trend_up.values if "trend_up" in d else d.up20d.values
    jst = d.index + pd.Timedelta(hours=10)
    hour, wd = jst.hour.values, jst.weekday.values
    out, last_day = [], None
    for i in range(500, len(d) - 2):
        m1, m3, m24 = r[i], c[i] - c[i - 3], c[i] - c[i - 24]
        z = max(abs(m1) / sd[i], abs(m3) / (sd[i] * np.sqrt(3)), abs(m24) / (sd[i] * np.sqrt(24)))
        if not (z >= z_th or abs(m24) >= yen):
            continue
        if window and not (hour[i] >= 10 or hour[i] < 1):
            continue
        if skip_mf and wd[i] in (0, 4):
            continue
        day = d.index[i].date()
        if day == last_day:
            continue
        side = 1 if (m3 if m3 != 0 else m24) > 0 else -1
        if aligned and (side == 1) != bool(up[i]):
            continue
        stop = c[i] - side * stop_atr * A[i]
        pips, _, _ = rs_.outcome(d, i + 1, side, stop, c[i] + side * tgt_atr * A[i], hold)
        out.append((d.index[i], side, pips, abs(d.Open.iat[i + 1] - stop) * 100))
        last_day = day
    return pd.DataFrame(out, columns=["time", "side", "pips", "risk"])
