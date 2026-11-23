"""担当AI (レンジ・トレンド・急変) と相場分析AI (相場判定・AIモデル) の「反省」。
23年分で各担当の判断の当たり外れを年代ごとに調べ、外れ方の傾向から改善案を作って、同じ物差しで比べる。
判断の物差し: 判断した向きに、その後24時間のうちに「損切り幅相当 (0.6ATR x 4) より先に、利確幅相当 (同じ幅) まで進んだか」。
(一方向に先に届いた方を勝ちとする。どちらにも届かなければ引き分けで数えない)"""
import numpy as np
import pandas as pd

from . import edge, specialists as sp
from .sysearch import ERAS


def first_touch(h, i, side, width, hours=24):
    H, L = h.High.values, h.Low.values
    e = h.Close.iat[i]
    up, dn = e + width, e - width
    for k in range(i + 1, min(i + 1 + hours, len(h))):
        hit_up, hit_dn = H[k] >= up, L[k] <= dn
        if hit_up and hit_dn:
            return 0
        if hit_up:
            return 1 if side == 1 else -1
        if hit_dn:
            return 1 if side == -1 else -1
    return 0


def era_of(t):
    for name, a, b in ERAS:
        if a <= t.year <= b:
            return name
    return None


def run(h: pd.DataFrame, step=6) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    c = h.Close
    k = 100 * (c - h.Low.rolling(14).min()) / (h.High.rolling(14).max() - h.Low.rolling(14).min())
    stoch = k.rolling(3).mean().values
    up20 = h.trend_up.values
    c4 = c.resample("4h").last().dropna()
    up4 = (c4.ewm(span=20).mean() > c4.ewm(span=50).mean()).shift(1).reindex(h.index, method="ffill").fillna(False).values
    adx = h.adx.values
    for i in range(1500, len(h) - 25, step):
        sub = h.iloc[i - 1200:i + 1]
        width = 0.6 * h.atr.iat[i] * 4
        era = era_of(h.index[i])
        reg = h.regime.iat[i]
        # --- レンジ担当: 今のルール と 改善案
        rv = sp.range_view(sub, h.rsi.iat[i])
        if rv["bias"] != 0:
            rows.append(("レンジ担当 (今)", era, reg, first_touch(h, i, rv["bias"], width)))
        # 改善案A: 3回以上反発した下値・上値の近く + ストキャスの行き過ぎ
        lines = rv["lines"][0]
        n_sup = int(lines.split("(")[1].split("回")[0]) if "(" in lines else 0
        n_res = int(lines.split("(")[2].split("回")[0]) if lines.count("(") > 1 else 0
        if rv["pos"] <= 0.25 and n_sup >= 3 and stoch[i] < 30:
            rows.append(("レンジ担当 改善A (3回反発+売られすぎ)", era, reg, first_touch(h, i, 1, width)))
        if rv["pos"] >= 0.75 and n_res >= 3 and stoch[i] > 70:
            rows.append(("レンジ担当 改善A (3回反発+売られすぎ)", era, reg, first_touch(h, i, -1, width)))
        # 改善案B: Aに加えて ADX が低い (トレンドが弱い) 時だけ
        if adx[i] < 20 and ((rv["pos"] <= 0.25 and n_sup >= 3 and stoch[i] < 30) or (rv["pos"] >= 0.75 and n_res >= 3 and stoch[i] > 70)):
            rows.append(("レンジ担当 改善B (Aに加えて流れが弱い時だけ)", era, reg, first_touch(h, i, 1 if rv["pos"] <= 0.25 else -1, width)))
        # --- トレンド担当: 今のルール と 改善案
        tv = sp.trend_view(sub)
        if tv["bias"] != 0:
            rows.append(("トレンド担当 (今)", era, reg, first_touch(h, i, tv["bias"], width)))
        if tv["status"] == "継続":
            side = 1 if tv["direction"] == "上昇" else -1
            # 改善案A: 20日平均と4時間足も同じ向きの時だけ
            if (up20[i] == (side == 1)) and (up4[i] == (side == 1)):
                rows.append(("トレンド担当 改善A (日足・4時間足もそろう)", era, reg, first_touch(h, i, side, width)))
                # 改善案B: さらに ADX>25 (流れが強い)
                if adx[i] > 25:
                    rows.append(("トレンド担当 改善B (Aに加えて流れが強い)", era, reg, first_touch(h, i, side, width)))
        if tv["status"] == "転換警戒":
            side = -1 if tv["direction"] == "上昇" else 1
            rows.append(("トレンド担当 転換警戒の逆張り", era, reg, first_touch(h, i, side, width)))
    # --- 急変担当: 急変の後に続くか戻るか (毎時)
    for i in range(1500, len(h) - 25):
        a = edge.anomaly(h.iloc[i - 600:i + 1])
        if a.get("level") != "急変":
            continue
        width = 0.6 * h.atr.iat[i] * 4
        d = 1 if a["move"] > 0 else -1
        era = era_of(h.index[i])
        rows.append(("急変担当: 急変と同じ向きに付いていく", era, "荒れ", first_touch(h, i, d, width)))
        rows.append(("急変担当: 急変と逆 (戻り) を狙う", era, "荒れ", first_touch(h, i, -d, width)))
        aligned = up20[i] == (-d == 1)
        if aligned:
            rows.append(("急変担当 改善 (戻りが大きな流れの方向の時だけ)", era, "荒れ", first_touch(h, i, -d, width)))
    r = pd.DataFrame(rows, columns=["担当", "年代", "相場状態", "結果"])
    return r, summarize(r)


def summarize(r: pd.DataFrame) -> list[str]:
    x = r[r.結果 != 0].copy()
    x["勝ち"] = x.結果 > 0
    t = x.groupby(["担当", "年代"], observed=True).勝ち.mean().unstack("年代")
    n = x.groupby("担当").size()
    t["全体"] = x.groupby("担当").勝ち.mean()
    t["回数"] = n
    t["50%を超えた年代"] = (t[[e[0] for e in ERAS if e[0] in t.columns]] > 0.5).sum(axis=1).astype(str) + "/" + str(len(ERAS))
    t = t.sort_index()
    fmt = t.copy()
    for col in fmt.columns:
        if col not in ("回数", "50%を超えた年代"):
            fmt[col] = (fmt[col] * 100).round(0).astype("Int64").astype(str) + "%"
    return ["判断した向きに、損切り幅相当より先に利確幅相当まで進んだ割合 (50%なら当てずっぽう)", "", fmt.to_markdown()]
