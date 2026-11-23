"""利用者のレンジ戦略を忠実に再現して検証する (USD/JPY 1時間足)

例: レンジ 159.2〜160.0 → 159.2 付近 (下値+3pips) に買い指値、159.0 (下値-20pips) を割ったら即損切り、
    上値の手前 (上値-10pips) で利確。損小利大で、勝率25%前後でもプラスになる設計。
もう一つ: 下値を割った後に元のレンジへ戻ってきた (下値を終値で回復した) ところを買う「戻り買い」。
"""
import numpy as np
import pandas as pd

from . import strategies as st

PIP = 0.01


def find_range(d, i, n, min_w, max_w, touches):
    """直前 n 本の高値/安値をレンジとし、幅と上下それぞれの反発回数で本物のレンジか判定する"""
    w = d.iloc[i - n:i]
    s, r = w.Low.min(), w.High.max()
    width = (r - s) / PIP
    if not (min_w <= width <= max_w):
        return None
    zone = 10 * PIP
    # 下値ゾーン/上値ゾーンに触れた「山・谷」の数 (6本以上離れたものを別の回として数える)
    def count(mask):
        idx = np.flatnonzero(mask.values); c, last = 0, -99
        for k in idx:
            if k - last >= 6: c += 1
            last = k
        return c
    if count(w.Low <= s + zone) < touches or count(w.High >= r - zone) < touches:
        return None
    return s, r


def simulate(d, n=96, min_w=40, max_w=150, touches=2, entry_off=3, stop_off=20, tgt_off=10,
             max_hold=120, trend=False, cost=0.5, mode="range"):
    O, H, L, C = d.Open.values, d.High.values, d.Low.values, d.Close.values
    up = (d.Close.ewm(span=480).mean().diff(24) > 0).values
    trades, i, N = [], n + 1, len(d)
    while i < N - 2:
        rg = find_range(d, i, n, min_w, max_w, touches) if mode == "range" else None
        if mode == "range":
            if rg is None or (trend and not up[i]):
                i += 1; continue
            s, r = rg
            entry, stop, tgt = s + entry_off * PIP, s - stop_off * PIP, r - tgt_off * PIP
            if not (L[i] <= entry and O[i] > entry):   # この足で下値付近まで下がって指値が約定したか
                i += 1; continue
            if O[i] <= stop:
                i += 1; continue
            j0 = i
            if L[i] <= stop:  # 約定した同じ足で損切りまで届いた
                trades.append(dict(t=d.index[i], pips=(stop - entry) / PIP - cost, why="損切り")); i += 1; continue
        else:  # 戻り買い: 下値を20pips以上割り込んだ後、48本以内に終値で下値を回復
            rg = find_range(d, i - 48, n, min_w, max_w, touches) if i > n + 49 else None
            if rg is None or (trend and not up[i]):
                i += 1; continue
            s, r = rg
            seg = slice(i - 48, i)
            low_since = L[seg].min()
            if not (low_since <= s - 20 * PIP and C[i] > s and C[i - 1] <= s and H[seg].max() < r):
                i += 1; continue
            entry, stop, tgt = O[i + 1], low_since - 5 * PIP, r - tgt_off * PIP
            if not (stop < entry < tgt):
                i += 1; continue
            j0 = i + 1
        px, why, k = None, "", j0
        for k in range(j0 + (1 if mode == "range" else 0), min(j0 + max_hold, N)):
            if O[k] <= stop: px, why = O[k], "損切り"
            elif L[k] <= stop: px, why = stop, "損切り"
            elif H[k] >= tgt: px, why = max(O[k], tgt) if O[k] > tgt else tgt, "利確"
            if px is not None: break
        if px is None:
            px, why = C[k], "時間切れ"
        trades.append(dict(t=d.index[j0], pips=(px - entry) / PIP - cost, why=why, risk=(entry - stop) / PIP))
        i = k + 1
    return pd.DataFrame(trades)


def report():
    raw = st.load()
    half = raw.index[len(raw) // 2]
    rows = []
    def add(name, t):
        if t.empty:
            rows.append({"作戦": name, "回数": 0}); return
        w, l = t[t.pips > 0], t[t.pips <= 0]
        eq = t.pips.cumsum()
        rows.append({"作戦": name, "回数": len(t), "勝率": f"{len(w)/len(t):.0%}",
                     "平均利益": round(w.pips.mean(), 1) if len(w) else 0, "平均損失": round(l.pips.mean(), 1) if len(l) else 0,
                     "PF": round(w.pips.sum() / -l.pips.sum(), 2) if l.pips.sum() < 0 else None,
                     "合計pips": round(t.pips.sum()), "最大DD": round((eq.cummax() - eq).max()),
                     "前半": round(t[t.t < half].pips.sum()), "後半": round(t[t.t >= half].pips.sum())})
    add("U1 ご提案どおり (損切り20/利確 上値-10)", simulate(raw))
    for so in (10, 15, 30):
        add(f"U1 損切り{so}pips", simulate(raw, stop_off=so))
    for n in (48, 72, 120):
        add(f"U1 レンジ期間{n}時間", simulate(raw, n=n))
    add("U1 反発3回以上のレンジだけ", simulate(raw, touches=3))
    add("U1 コスト1.0pips", simulate(raw, cost=1.0))
    add("U2 U1+大きな流れが上向きの時だけ", simulate(raw, trend=True))
    add("U3 下値割れからの戻り買い", simulate(raw, mode="recover"))
    add("U3 +大きな流れが上向きの時だけ", simulate(raw, mode="recover", trend=True))
    pd.set_option("display.width", 250)
    print(f"USD/JPY 1時間足 {raw.index[0]:%Y-%m-%d}〜{raw.index[-1]:%Y-%m-%d} / 前半後半の境 {half:%Y-%m-%d}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    report()
