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


START = pd.Timestamp("2026-11-25", tz="UTC")   # ルールの凍結 (影の運用の始まり)


def check(h: pd.DataFrame) -> list[dict]:
    """影の作戦の取引を記録する (通知しない)。
    2026-11-27 の点検で発覚: 以前は「最新の足の合図」だけを探していたが、lab の関数は決済まで終わった取引しか返さないので、
    一度も記録されていなかった。→ 直近900本で終わった取引を、凍結日以降の分だけ、まだ無いものを足す (結果の R つき)。
    合図は過去の値動きだけで決まるので、後から記録しても未来の情報は混ざらない"""
    d = lab.prep(h.iloc[-900:])
    if "trend_up" not in d:
        d["trend_up"] = d.Close.ewm(span=480).mean().diff(24) > 0
    found = []
    for name, c in CANDIDATES.items():
        t = c["fn"](d, **c["params"])
        if not len(t):
            continue
        t = t[t.time >= START]
        for r in t.itertuples():
            found.append({"time_utc": f"{r.time:%Y-%m-%d %H:%M}", "strategy": name, "side": int(r.side),
                          "entry": round(float(d.Open.asof(r.time)), 3), "risk_pips": round(float(r.risk), 1),
                          "exit_utc": f"{r.exit_time:%Y-%m-%d %H:%M}" if hasattr(r, "exit_time") else "",
                          "R": round(float(r.R), 3) if hasattr(r, "R") else round(float(r.pips) / float(r.risk), 3) if hasattr(r, "pips") and r.risk else ""})
    new = pd.DataFrame(found)
    if len(new):
        old = pd.read_csv(PATH) if os.path.exists(PATH) else pd.DataFrame(columns=new.columns)
        key = set(zip(old.get("strategy", []), old.get("time_utc", [])))
        new = new[[(a, b) not in key for a, b in zip(new.strategy, new.time_utc)]]
        if len(new):
            os.makedirs(os.path.dirname(PATH), exist_ok=True)
            pd.concat([old, new]).to_csv(PATH, index=False)
    return new.to_dict("records") if len(new) else []
