"""リスク管理部隊: 今、取引してよい状況かを判定する (向きは出さない)。
部下: 指標警戒AI / 値動き異常AI / 週末・時間帯AI / 介入警戒AI / 連敗監視AI
結論: 通常 / 注意 / 警戒 (警戒は「新しく入らない方がよい」)"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def check(h: pd.DataFrame, ops: list[dict] | None = None, now=None) -> dict:
    now = now or pd.Timestamp.now(tz="UTC")
    items = []   # (重さ 1=注意 2=警戒, 担当, 理由)
    # 指標警戒AI
    try:
        from .. import calendar as cal
        for e in cal.upcoming(now, 24, impacts=("High",)):
            mins = (e["time"] - now).total_seconds() / 60
            items.append((2 if mins <= 60 else 1, "指標警戒", f"{cal.fmt(e, False)} まで{mins / 60:.1f}時間"))
            break
    except Exception:
        pass
    # 値動き異常AI
    c = h.Close
    tr = pd.concat([h.High - h.Low, (h.High - c.shift()).abs(), (h.Low - c.shift()).abs()], axis=1).max(axis=1)
    ratio = tr.rolling(3).mean().iat[-1] / tr.rolling(480, min_periods=100).mean().iat[-1]
    if ratio >= 2.5:
        items.append((2, "値動き異常", f"直近の値動きが普段の{ratio:.1f}倍 (急変の最中)"))
    elif ratio >= 1.6:
        items.append((1, "値動き異常", f"直近の値動きが普段の{ratio:.1f}倍"))
    # 週末・時間帯AI
    jst = now + pd.Timedelta(hours=9)
    if (jst.weekday() == 4 and jst.hour >= 23) or jst.weekday() == 5:
        items.append((1, "週末・時間帯", "週末前 (持ち越しで窓が開くことがある)"))
    ny = now.tz_convert("America/New_York")
    if 16 <= ny.hour < 19:
        items.append((1, "週末・時間帯", "スプレッドが広がりやすい時間帯 (NY 16〜19時)"))
    # 介入警戒AI: 見出しと価格水準
    notes = " ".join(o.get("reason", "") for o in (ops or []))
    if "介入" in notes:
        items.append((1, "介入警戒", "ニュースに「介入」の言葉 (急な円高に注意)"))
    # 円安けん制の段階 (ルールの読み取りと Gemini の高い方)。前回の記録より強まっていたら知らせる
    stage = max([int(o.get("stage", 0)) for o in (ops or []) if "stage" in o] or [0])
    prev = 0
    try:
        log = pd.read_csv(os.path.join(ROOT, "journal", "org.csv"))
        if "介入けん制の段階" in log:
            prev = int(log["介入けん制の段階"].dropna().iloc[-24:].max())
    except Exception:
        pass
    if stage >= 3:
        items.append((2 if stage >= 4 else 1, "介入警戒", f"円安けん制が段階{stage} (強い言葉)。急な円高に注意" + (" — 強まっている" if stage > prev else "")))
    elif stage > prev and stage >= 2:
        items.append((1, "介入警戒", f"円安けん制が段階{prev}→{stage}に強まった"))
    hi20 = h.High.iloc[-480:].max()
    if c.iat[-1] >= 155 and c.iat[-1] >= hi20 - 0.3:
        items.append((1, "介入警戒", f"20日の高値圏 ({hi20:.2f}) で円安が進んでいる"))
    # 連敗監視AI: 送ったシグナルの直近の結果
    try:
        from ..periodic import evaluate_signals
        ev = evaluate_signals(h)
        if len(ev):
            sent = ev[(ev.sent.astype(str) == "True") & ev.pips.notna()].tail(10)
            streak = 0
            for p in sent.pips[::-1]:
                if p < 0:
                    streak += 1
                else:
                    break
            if streak >= 4:
                items.append((1, "連敗監視", f"送ったシグナルが{streak}連敗中 (相場が合っていない可能性)"))
    except Exception:
        pass
    # データ監視部隊・資金管理部隊の停止は「警戒」
    try:
        from . import health, money
        for x in health.check(h, now)["problems"]:
            items.append((2, "データ監視", x))
        lm = money.limits(h, now)
        if lm["stop"]:
            items.append((2, "資金管理", lm["text"]))
    except Exception:
        pass
    lvl = max([w for w, _, _ in items], default=0)
    return {"level": ["通常", "注意", "警戒"][lvl], "items": [(who, why) for _, who, why in items]}
