"""全作戦を検証して表で出す: python -m fxpred strategies"""
import numpy as np
import pandas as pd

from . import strategies as st

PLANS = {
    "A レンジ逆張り(買いのみ) ※ご提案": dict(rule=st.range_reversal),
    "A2 レンジ逆張り(売買両方)": dict(rule=st.range_reversal, both_sides=True),
    "A3 レンジ逆張り+材料待ち限定+材料前決済": dict(rule=st.range_reversal, event_wait=True, exit_before_event=True),
    "A4 A3を売買両方": dict(rule=st.range_reversal, both_sides=True, event_wait=True, exit_before_event=True),
    "B ブレイクアウト(順張り)": dict(rule=st.breakout),
    "C 押し目買い/戻り売り": dict(rule=st.trend_pullback),
    "D ボリンジャー逆張り": dict(rule=st.bollinger_reversion),
}


def run(cost=0.5):
    raw = st.load()
    d = st.indicators(raw)
    half = d.index[len(d) // 2]
    rows, trades = [], {}
    for name, p in PLANS.items():
        p = dict(p); rule = p.pop("rule")
        t = st.simulate(d, rule, cost_pips=cost, **p)
        trades[name] = t
        a = st.stats(t[t.entry_time < half]) if len(t) else {}
        b = st.stats(t[t.entry_time >= half]) if len(t) else {}
        rows.append({"作戦": name, **st.stats(t), "前半合計": a.get("合計pips", 0), "後半合計": b.get("合計pips", 0)})
    print(f"USD/JPY 1時間足 {d.index[0]:%Y-%m-%d}〜{d.index[-1]:%Y-%m-%d} / 往復コスト{cost}pips / 前半後半の境 {half:%Y-%m-%d}")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(pd.DataFrame(rows).to_string(index=False))
    return d, trades


if __name__ == "__main__":
    run()
