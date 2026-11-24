"""資金管理部隊: どれだけ張るか、どこで止めるか。
- 1回の損失の上限: 資金の RISK_PCT % (検証の推奨は1〜3%。最大下落20%までに収めるなら1%前後)。損切り幅から数量を計算できる
- 1日・1週間の損失の上限: 送った合図の実際の結果 (損切り1回分 = 1R) で数え、日 -DAY_R R・週 -WEEK_R R に達したら新しく入らない
- 信頼度に応じた調整: 信頼度が低い合図 (65%未満で送る損小利大の形など) は量を半分
※ 利用者の指示で、通知には数量を載せない。ここでは止める判断にだけ使い、数量は聞かれた時に出す"""
import os

import numpy as np
import pandas as pd

RISK_PCT = 1.0
DAY_R, WEEK_R = 2.0, 4.0


def size(capital_yen: float, entry: float, stop: float, conf: int, risk_pct=RISK_PCT) -> dict:
    """数量の計算 (1万通貨単位)。信頼度65%未満は半分"""
    risk_yen = capital_yen * risk_pct / 100 * (0.5 if conf < 65 else 1.0)
    per_unit = abs(entry - stop) * 10000          # 1万通貨で損切りまでに失う円
    lots = np.floor(risk_yen / per_unit * 10) / 10 if per_unit > 0 else 0
    lev = lots * 10000 * entry / capital_yen
    return {"lots_10k": float(lots), "risk_yen": round(risk_yen), "leverage": round(lev, 1), "capped": lev > 25}


def results(h: pd.DataFrame) -> pd.DataFrame:
    """送った合図の結果を R (損切り幅=1) で"""
    from ..periodic import evaluate_signals
    ev = evaluate_signals(h)
    if ev.empty:
        return ev
    ev = ev[(ev.sent.astype(str) == "True") & ev.pips.notna()].copy()
    risk = (ev.entry.astype(float) - ev.stop.astype(float)).abs() * 100
    ev["R"] = ev.pips / risk.replace(0, np.nan)
    ev["t"] = pd.to_datetime(ev.time_utc, utc=True)
    return ev


def limits(h: pd.DataFrame, now=None) -> dict:
    now = now or pd.Timestamp.now(tz="UTC")
    try:
        ev = results(h)
    except Exception:
        return {"stop": False, "text": "結果を数えられず"}
    if ev.empty:
        return {"stop": False, "text": "送った合図の結果なし"}
    jst = now + pd.Timedelta(hours=9)
    day0 = (jst.normalize() - pd.Timedelta(hours=9))
    week0 = day0 - pd.Timedelta(days=jst.weekday())
    d, w = ev.R[ev.t >= day0].sum(), ev.R[ev.t >= week0].sum()
    stop = d <= -DAY_R or w <= -WEEK_R
    txt = f"今日 {d:+.1f}R / 今週 {w:+.1f}R (上限 日-{DAY_R:.0f}R・週-{WEEK_R:.0f}R)"
    return {"stop": bool(stop), "text": ("損失の上限に達したため新しく入らない: " if stop else "") + txt}
