"""指示されたときに相場を分析し、診断材料 (テクニカル・AIモデル・マクロ市場) をまとめる。
最終的な 買い/売り/HOLD と理由は、Claude がこの材料と最新ニュース・経済指標予定を合わせて判断し、
`record` で記録する (後で `evaluate` が答え合わせする)。"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import data, features as ft, macro, model
from .config import PAIRS

JOURNAL = os.getenv("FX_JOURNAL", "journal/diagnoses.csv")
NAME2TICKER = {v: k for k, v in PAIRS.items()}


def _r(x: float) -> float:
    return float(f"{x:.5g}")


def _rsi(c, n=14):
    return float(ft._rsi(c, n).iloc[-1])


def _tf(c: pd.Series, rule: str) -> pd.Series:
    return c.resample(rule, label="right", closed="right").last().dropna() if rule != "15min" else c


def technical(df: pd.DataFrame) -> dict:
    c = df["Close"]
    out, score, reasons = {}, 0, []
    for label, rule in (("15分足", "15min"), ("1時間足", "1h"), ("4時間足", "4h")):
        s = _tf(c, rule)
        e20, e50 = s.ewm(span=20).mean().iloc[-1], s.ewm(span=50).mean().iloc[-1]
        macd = s.ewm(span=12).mean() - s.ewm(span=26).mean()
        hist = float((macd - macd.ewm(span=9).mean()).iloc[-1])
        trend = "上昇" if s.iloc[-1] > e20 > e50 else "下降" if s.iloc[-1] < e20 < e50 else "もみ合い"
        rsi = _rsi(s)
        out[label] = dict(trend=trend, rsi=round(rsi, 1), macd_hist="プラス" if hist > 0 else "マイナス")
        w = 2 if label == "4時間足" else 1
        score += w * (1 if trend == "上昇" else -1 if trend == "下降" else 0)
        if trend != "もみ合い":
            reasons.append(f"{label}は{trend}トレンド (EMA20/50の並び)")
        if rsi >= 70:
            score -= 1; reasons.append(f"{label}RSI {rsi:.0f} で買われすぎ")
        elif rsi <= 30:
            score += 1; reasons.append(f"{label}RSI {rsi:.0f} で売られすぎ")
    last = c.iloc[-1]
    day, week = df.iloc[-96:], df.iloc[-96 * 5:]
    m20, s20 = c.rolling(80).mean().iloc[-1], c.rolling(80).std().iloc[-1]  # 1時間足20本相当
    out["価格"] = _r(last)
    out["24時間 高値/安値"] = [_r(day.High.max()), _r(day.Low.min())]
    out["5日 高値/安値"] = [_r(week.High.max()), _r(week.Low.min())]
    out["ボリンジャー位置(σ)"] = round(float((last - m20) / s20), 2)
    atr = (df.High - df.Low).rolling(16).mean().iloc[-1]
    out["4時間の平均変動幅(目安)"] = _r(atr * 4)
    out["テクニカルスコア"] = int(score)  # -6〜+6 程度。プラスは買い寄り
    out["テクニカル根拠"] = reasons
    return out


def macro_snapshot() -> dict:
    m = macro.fetch("5d", "15m").ffill()
    snap = {}
    for t, name in macro.MACRO.items():
        s = m[t].dropna()
        if s.empty:
            continue
        now, h4, d1 = s.iloc[-1], s.iloc[-17] if len(s) > 17 else s.iloc[0], s.iloc[-97] if len(s) > 97 else s.iloc[0]
        if t in macro.LEVEL:
            snap[name] = f"{now:.2f} (4時間 {now - h4:+.2f} / 1日 {now - d1:+.2f})"
        else:
            snap[name] = f"{now:.2f} (4時間 {np.log(now / h4) * 100:+.2f}% / 1日 {np.log(now / d1) * 100:+.2f}%)"
    return snap


def run(pairs: list[str] | None) -> dict:
    d = data.fetch()
    f = ft.build_all(d)
    m = model.train(f)
    latest = f.groupby("ticker").tail(1)
    p_up = dict(zip(latest.ticker, model.predict_proba(m, latest).values))
    tickers = [NAME2TICKER[p] for p in pairs] if pairs else list(PAIRS)
    report = {"分析時刻(UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
              "マクロ市場": macro_snapshot(), "通貨ペア": {}}
    for t in tickers:
        tech = technical(d[t])
        tech["AIモデル 1時間後の上昇確率"] = f"{p_up[t]:.0%}"
        tech["最新足"] = f"{d[t].index[-1]:%m/%d %H:%M} UTC"
        report["通貨ペア"][PAIRS[t]] = tech
    report["検証済みの信頼度"] = ("過去検証では テクニカル・AIモデル・マクロ指標いずれも方向の正解率は約50〜53%で、"
                          "取引コスト込みの優位性は未確認。単独の根拠にしないこと。")
    return report


def record(pair: str, verdict: str, reason: str, horizon_h: float, price: float | None = None) -> dict:
    t = NAME2TICKER[pair]
    if price is None:
        price = _r(data.fetch()[t]["Close"].iloc[-1])
    row = dict(time=datetime.now(timezone.utc).isoformat(timespec="minutes"), pair=pair,
               verdict=verdict.upper(), price=price, horizon_h=horizon_h, reason=reason)
    os.makedirs(os.path.dirname(JOURNAL), exist_ok=True)
    pd.DataFrame([row]).to_csv(JOURNAL, mode="a", header=not os.path.exists(JOURNAL), index=False)
    return row


def evaluate() -> str:
    """過去の診断を答え合わせする。HOLD は「その後の値動きが4時間平均変動幅の半分未満」なら正解"""
    if not os.path.exists(JOURNAL):
        return "まだ診断記録がありません"
    j = pd.read_csv(JOURNAL, parse_dates=["time"])
    d = data.fetch()
    rows = []
    for _, r in j.iterrows():
        t = NAME2TICKER[r.pair]
        px = d[t]["Close"]
        end = r.time + pd.Timedelta(hours=r.horizon_h)
        if px.index[-1] < end:
            rows.append({**r, "結果": "判定待ち"}); continue
        after = float(px[px.index <= end].iloc[-1])
        move = (after - r.price) / (0.01 if "JPY" in r.pair else 0.0001)
        typical = float((d[t].High - d[t].Low).rolling(16).mean().loc[:r.time].iloc[-1]) * 4 / (0.01 if "JPY" in r.pair else 0.0001)
        ok = {"BUY": move > 0, "SELL": move < 0, "HOLD": abs(move) < typical / 2}[r.verdict]
        rows.append({**r, "その後(pips)": round(move, 1), "結果": "正解" if ok else "不正解"})
    e = pd.DataFrame(rows)
    done = e[e["結果"] != "判定待ち"]
    head = (f"診断 {len(e)}件 / 判定済み {len(done)}件 / 正解率 {(done['結果'] == '正解').mean():.0%}"
            if len(done) else f"診断 {len(e)}件 / すべて判定待ち")
    by = done.groupby("verdict")["結果"].apply(lambda s: f"{(s == '正解').mean():.0%} ({len(s)}件)").to_dict() if len(done) else {}
    cols = [c for c in ["time", "pair", "verdict", "price", "horizon_h", "その後(pips)", "結果"] if c in e]
    return head + ("\n種類別: " + json.dumps(by, ensure_ascii=False) if by else "") + "\n" + e[cols].tail(20).to_string(index=False)
