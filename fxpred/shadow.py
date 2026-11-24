"""影の運用: 新しい作戦を通知せずに記録だけして、実際の相場での成績を今のルールと比べる。
採用するかは、この記録と20年の研究の両方で効果を確かめてから決める (CHANGELOG に残す)。
記録: journal/shadow.csv (シグナル) / 成績は週次・月次の見直しで集計"""
import os

import pandas as pd

from . import lab

PATH = os.path.join(os.path.dirname(__file__), "..", "journal", "shadow.csv")
CANDIDATES = {
    # 研究室 (reports/lab.md) で6期間すべてプラスだった設定のまとまりの中心
    "レンジ反発 (2日・3回反発・利確6割)": dict(fn=lab.range_bounce, params=dict(n=48, touches=3, osc_lv=30, need_osc=False,
                                                                        tgt_frac=0.6, stop_atr=0.45, trend_only=False)),
    # 23年の研究 (reports/overnight.md) で7年代すべてプラスだった中心の作戦
    "急変に付いていく (3.5倍・流れ方向・月金除外)": dict(fn=lab.shock_follow, params=dict(z_th=3.5)),
    "急変に付いていく 損小利大 (4倍・損切1ATR/利確4ATR・流れ方向)": dict(fn=lab.shock_follow, params=dict(z_th=4, skip_mf=False, stop_atr=1.0, tgt_atr=4.0)),
    # 時間帯の癖を狙う担当: 東京の朝 (8〜10時) の値幅を流れの方向に抜けたら。23年で年103回・作る +0.10R / 確かめ +0.06R・78%の年でプラス
    "東京朝の値幅ブレイク (流れの方向・16時間)": dict(fn=lab.tokyo_morning_break, params=dict(hold=16)),
}


def check(h: pd.DataFrame) -> list[dict]:
    """最新の確定足で影の作戦のシグナルが出ていれば記録する"""
    d = lab.prep(h.iloc[-900:])
    if "trend_up" not in d:
        d["trend_up"] = d.Close.ewm(span=480).mean().diff(24) > 0
    last = d.index[-2]   # range_bounce は次の足の始値で入るので、1本前の確定足までを判定対象にする
    found = []
    for name, c in CANDIDATES.items():
        t = c["fn"](d, **c["params"])
        t = t[t.time == last] if len(t) else t
        for r in t.itertuples():
            found.append({"time_utc": f"{r.time:%Y-%m-%d %H:%M}", "strategy": name, "side": int(r.side),
                          "entry": round(float(d.Open.iat[-1]), 3), "risk_pips": round(float(r.risk), 1)})
    if found:
        old = pd.read_csv(PATH) if os.path.exists(PATH) else pd.DataFrame()
        new = pd.DataFrame(found)
        if len(old):
            new = new[~new.time_utc.isin(old.time_utc[old.strategy.isin(new.strategy)])]
        if len(new):
            os.makedirs(os.path.dirname(PATH), exist_ok=True)
            new.to_csv(PATH, mode="a", header=not os.path.exists(PATH), index=False)
    return found
