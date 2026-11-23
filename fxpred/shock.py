"""急落 (介入・指標・リスク回避) の後の「戻り」を狙う作戦の検証 (USD/JPY 1時間足)

急落の定義: 直近 lookback 時間の高値から drop 円以上下げた
入り方:
  A 下げ止まり確認: 急落後、安値更新が止まって陽線で引けた足の次で買う
  B 即買い: 急落を検知した足の次で買う (落ちるナイフをつかむ)
損切り: 急落の安値 - buffer 円 (B は 検知時の値 - stop_yen)
利確:   下げ幅の retrace 割合だけ戻ったところ。届かなければ max_hold 時間で手仕舞い
急騰 (円安方向) の後の反落を売る場合も対称に検証する
"""
import numpy as np
import pandas as pd

from . import strategies as st
from .portfolio import MAX_LEV

PIP = 0.01


def find(d, drop=2.0, lookback=24, entry="A", retrace=0.3, buffer=0.2, stop_yen=1.0, max_hold=72,
         side=1, wait=12, cost=0.01):
    O, H, L, C = d.Open.values, d.High.values, d.Low.values, d.Close.values
    out, i, n = [], lookback, len(d)
    while i < n - 2:
        if side == 1:
            top = H[i - lookback:i + 1].max(); mv = top - C[i]
        else:
            top = L[i - lookback:i + 1].min(); mv = C[i] - top
        if mv < drop:
            i += 1; continue
        # 急落を検知
        if entry == "B":
            j = i + 1; e = O[j]
            stop = e - side * stop_yen
            extreme = L[i] if side == 1 else H[i]
        else:  # 下げ止まり (陽線) を wait 時間以内に待つ
            j = None
            ext = L[i] if side == 1 else H[i]
            for k in range(i + 1, min(i + wait, n - 1)):
                ext = min(ext, L[k]) if side == 1 else max(ext, H[k])
                turned = (C[k] > O[k] and L[k] > ext - 1e-9 and L[k] > L[k - 1]) if side == 1 else \
                         (C[k] < O[k] and H[k] < H[k - 1])
                if turned:
                    j = k + 1; break
            if j is None:
                i += wait; continue
            e = O[j]; extreme = ext
            stop = extreme - side * buffer
        size = abs(top - extreme)
        tgt = extreme + side * retrace * size
        if side * (tgt - e) <= 0 or side * (e - stop) <= 0:
            i = j; continue
        px, why = None, ""
        for k in range(j, min(j + max_hold, n)):
            if side == 1:
                if O[k] <= stop: px, why = O[k], "損切り"
                elif L[k] <= stop: px, why = stop, "損切り"
                elif H[k] >= tgt: px, why = max(O[k], tgt), "利確"
            else:
                if O[k] >= stop: px, why = O[k], "損切り"
                elif H[k] >= stop: px, why = stop, "損切り"
                elif L[k] <= tgt: px, why = min(O[k], tgt), "利確"
            if px is not None: break
        if px is None: px, why = C[k], "時間切れ"
        out.append(dict(検知=d.index[i], 入り=d.index[j], 下げ幅=round(size, 2), 入値=round(e, 2), 損切り=round(stop, 2),
                        利確=round(tgt, 2), 決済=round(px, 2), 結果=why, 円=round(side * (px - e) - cost, 3),
                        R=(side * (px - e) - cost) / abs(e - stop), dist=abs(e - stop), price=e))
        i = k + 1 + 24  # 同じ急落で何度も入らない
    return pd.DataFrame(out)


def equity(t, risk, init=1_000_000):
    eq, peak, mdd = init, init, 0
    for R, dist, px in zip(t.R, t.dist, t.price):
        f = min(risk, MAX_LEV * dist / px)
        eq *= 1 + f * R; peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
    return eq, mdd


def run():
    d = st.load()
    print(f"USD/JPY 1時間足 {d.index[0]:%Y-%m-%d}〜{d.index[-1]:%Y-%m-%d}")
    rows = []
    for side in (1, -1):
        for drop in (1.5, 2.0, 3.0):
            for entry in ("A", "B"):
                for retrace in (0.25, 0.4):
                    t = find(d, drop=drop, entry=entry, retrace=retrace, side=side)
                    if t.empty:
                        continue
                    res = {"方向": "急落後の買い" if side == 1 else "急騰後の売り", "下げ幅": drop, "入り方": entry, "戻り目標": retrace,
                           "回数": len(t), "勝率": f"{(t.円 > 0).mean():.0%}", "平均R": round(t.R.mean(), 2),
                           "合計(円/1ドル)": round(t.円.sum(), 2)}
                    for r in (0.05, 0.10):
                        eq, mdd = equity(t, r)
                        res[f"リスク{r:.0%}: 100万円→"] = f"{eq / 1e4:.0f}万 (DD{mdd:.0%})"
                    rows.append(res)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    run()


def trend_aligned(d, drop=2.0, retrace=0.4, entry="A"):
    """採用候補: 大きな流れ (20日平均の向き) の方向へ戻る反動だけを取る。
    上昇トレンド中の急落 → 下げ止まりを確認して買い / 下降トレンド中の急騰 → 上げ止まりを確認して売り"""
    up = d.Close.ewm(span=480).mean().diff(24) > 0
    tb = find(d, drop=drop, entry=entry, retrace=retrace, side=1)
    ts = find(d, drop=drop, entry=entry, retrace=retrace, side=-1)
    tb = tb[up.reindex(tb.検知).values] if len(tb) else tb
    ts = ts[~up.reindex(ts.検知).values] if len(ts) else ts
    return pd.concat([tb, ts]).sort_values("入り").reset_index(drop=True)
