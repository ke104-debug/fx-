"""5分監視の補佐 (Groq 無料枠): 5分ごとに、直近1時間の1分足・下値/上値・最新の会議の結論・調査班の報告・2時間以内の指標を読ませ、
「今の状況・30分後の向きの見立て・注意・チャンス候補」を出させる。
- 急変の通知には、この状況判断を1行添える (原因と、続くか戻るかの手がかり)
- 見立ては journal/quick_ai.csv に記録し、週次で30分後・60分後の当たりを採点する。判断の重みは0 (ルールは変えない)
回数: 1日280回まで (5分監視は1日288回)。Groq の残りが2割を切ったら止まる (groq.allowed)"""
import json
import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
PATH = os.path.join(ROOT, "journal", "quick_ai.csv")


def read(bars, levels: dict, st: dict) -> dict | None:
    """bars: 1分足 (t, o, h, l, c)。st: 5分監視の状態 (1日の回数を数える)"""
    from . import groq
    day = time.strftime("%Y-%m-%d", time.gmtime())
    cnt = st.setdefault("groq_quick", {})
    if cnt.get("day") != day:
        cnt.clear(); cnt.update(day=day, n=0)
    if cnt["n"] >= 280 or not groq.allowed("5分監視"):
        return None
    c = bars[-1][4]
    ch = {k: (c - bars[-1 - k][4]) * 100 for k in (5, 15, 30, 60) if len(bars) > k}
    hi60, lo60 = max(b[2] for b in bars[-60:]), min(b[3] for b in bars[-60:])
    try:
        og = pd.read_csv(os.path.join(ROOT, "journal", "org.csv")).iloc[-1]
        org = f"{og.get('conclusion', '')} / 方向 {og.get('verdict', '')} {og.get('overall', 0):+.2f} / リスク {og.get('risk', '')}"
    except Exception:
        org = "記録なし"
    try:
        from . import research
        rr = research.recent(hours=3, n=3)
    except Exception:
        rr = []
    try:
        from .. import calendar as cal
        ev = " / ".join(cal.fmt(e, False) for e in cal.upcoming(pd.Timestamp.now(tz="UTC"), 2, impacts=("High", "Medium")))
    except Exception:
        ev = ""
    prompt = ("FX (ドル円) の5分監視の補佐。以下だけを根拠に、今の状況を短く判断する。推測で材料を足さない。JSON だけ: "
              '{"状況": "30字以内", "向き30分": -1〜1 (30分後に上がりそう=+), "注意": "30字以内 (無ければ空)", '
              '"チャンス": "なし" か "買い候補" か "売り候補", "確かさ": 0〜1}\n\n'
              f"今 {c:.3f}。動き: 5分 {ch.get(5, 0):+.1f} / 15分 {ch.get(15, 0):+.1f} / 30分 {ch.get(30, 0):+.1f} / 60分 {ch.get(60, 0):+.1f} pips。"
              f"直近60分の高値 {hi60:.3f} 安値 {lo60:.3f}\n"
              f"下値 {levels.get('support', '-')} / 上値 {levels.get('resistance', '-')} / 相場の状態 {levels.get('regime', '-')}\n"
              f"会議の結論: {org}\n2時間以内の指標: {ev or 'なし'}\n調査班 (直近3時間): " + (" / ".join(rr) if rr else "なし"))
    r = groq.call("5分監視", prompt)
    cnt["n"] += 1
    g = None
    if time.gmtime(bars[-1][0]).tm_min % 15 < 5:     # 15分ごとに Gemini にも同じことを聞く (別の目で見た見立て)
        from . import gem
        g = gem.call("15分の見立て", prompt)
    if not isinstance(r, dict) and isinstance(g, dict):
        r, g = g, None
    if not isinstance(r, dict):
        return None
    try:
        v = max(-1.0, min(1.0, float(r.get("向き30分", 0))))
        conf = max(0.0, min(1.0, float(r.get("確かさ", 0))))
    except Exception:
        v, conf = 0.0, 0.0
    row = {"time_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime(bars[-1][0])), "price": round(c, 3), "view30": round(v, 2),
           "conf": round(conf, 2), "chance": str(r.get("チャンス", ""))[:6], "state": str(r.get("状況", ""))[:40], "caution": str(r.get("注意", ""))[:40]}
    if isinstance(g, dict):
        try:
            row["g_view30"] = round(max(-1.0, min(1.0, float(g.get("向き30分", 0)))), 2)
            row["g_conf"] = round(max(0.0, min(1.0, float(g.get("確かさ", 0)))), 2)
        except Exception:
            pass
        row["g_state"] = str(g.get("状況", ""))[:40]
    try:
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        pd.DataFrame([row]).to_csv(PATH, mode="a", header=not os.path.exists(PATH), index=False)
    except Exception:
        pass
    return row


def score(h: pd.DataFrame) -> list[str]:
    """週次: 5分監視の補佐の30分後・60分後の向きの当たり (1時間足の終値で近似)"""
    if not os.path.exists(PATH):
        return ["5分監視の補佐: まだ記録なし"]
    d = pd.read_csv(PATH)
    d["t"] = pd.to_datetime(d.time_utc, utc=True)
    c = h.Close
    out = []
    for mins in (30, 60):
        fut = c.reindex(d.t + pd.Timedelta(minutes=mins), method="nearest", tolerance=pd.Timedelta(minutes=35)).values
        k = (d.view30.abs() >= 0.3) & pd.notna(fut)
        if k.sum() >= 10:
            hit = ((d.view30[k] > 0) == (fut[k] > d.price[k])).mean()
            out.append(f"5分監視の補佐: 向きを出した{int(k.sum())}回、{mins}分後の当たり {hit:.0%}")
    if "g_view30" in d:
        fut = c.reindex(d.t + pd.Timedelta(minutes=30), method="nearest", tolerance=pd.Timedelta(minutes=35)).values
        k = d.g_view30.notna() & (d.g_view30.abs() >= 0.3) & pd.notna(fut)
        if k.sum() >= 10:
            out.append(f"Gemini の15分の見立て: {int(k.sum())}回、30分後の当たり {((d.g_view30[k] > 0) == (fut[k] > d.price[k])).mean():.0%}")
        both = k & (d.view30.abs() >= 0.3) & (np.sign(d.view30) == np.sign(d.g_view30))
        if both.sum() >= 10:
            out.append(f"Groq と Gemini の向きが一致した時: {int(both.sum())}回、30分後の当たり {((d.view30[both] > 0) == (fut[both] > d.price[both])).mean():.0%}")
    ch = d[d.chance.astype(str).str.contains("候補")]
    if len(ch):
        out.append(f"チャンス候補を出した回数 {len(ch)} (確かさ0.7以上 {int((ch.conf >= 0.7).sum())})")
    return out or ["5分監視の補佐: 答え合わせできる記録がまだ少ない"]
