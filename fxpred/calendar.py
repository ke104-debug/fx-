"""Forex Factory の経済指標カレンダー (無料の公開フィード) を使う。
重要度は Forex Factory の色と同じ: 🔴 High (赤) / 🟠 Medium (オレンジ) / 🟡 Low (黄)。
フィードには予想・前回はあるが、発表後の結果 (実績) は含まれない。
取得は1時間に1回まで (state/ff.json にキャッシュ) にして、提供元の負荷に配慮する。"""
import json
import os
import time
import urllib.request

import pandas as pd

URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CACHE = os.getenv("FX_FF_CACHE", "state/ff.json")
EMOJI = {"High": "🔴", "Medium": "🟠", "Low": "🟡", "Holiday": "⚪"}
JA = {"Core PCE Price Index": "コアPCE物価指数", "PCE Price Index": "PCE物価指数", "GDP": "GDP", "GDP Price Index": "GDPデフレーター",
      "Unemployment Claims": "新規失業保険申請件数", "CB Consumer Confidence": "消費者信頼感指数",
      "Manufacturing PMI": "製造業PMI", "Services PMI": "サービス業PMI",
      "Non-Farm Employment Change": "雇用統計 (非農業部門雇用者数)", "Unemployment Rate": "失業率",
      "Average Hourly Earnings": "平均時給", "Core CPI": "コアCPI", "CPI": "消費者物価指数", "Core PPI": "コアPPI", "PPI": "生産者物価指数",
      "Retail Sales": "小売売上高", "Core Retail Sales": "コア小売売上高", "ISM Manufacturing PMI": "ISM製造業景況指数",
      "ISM Services PMI": "ISM非製造業景況指数", "Federal Funds Rate": "FOMC 政策金利", "FOMC Statement": "FOMC 声明",
      "FOMC Press Conference": "FOMC 議長会見", "FOMC Meeting Minutes": "FOMC 議事要旨", "BOJ Policy Rate": "日銀 政策金利",
      "Monetary Policy Statement": "金融政策声明", "BOJ Press Conference": "日銀総裁会見", "Tokyo Core CPI": "東京都区部コアCPI",
      "National Core CPI": "全国コアCPI", "JOLTS Job Openings": "JOLTS求人件数", "ADP Non-Farm Employment Change": "ADP雇用統計",
      "UoM Consumer Sentiment": "ミシガン大学消費者信頼感", "Durable Goods Orders": "耐久財受注",
      "Core Durable Goods Orders": "コア耐久財受注", "Empire State Manufacturing Index": "NY連銀製造業景況指数",
      "Philly Fed Manufacturing Index": "フィラデルフィア連銀景況指数", "Fed Chair Powell Speaks": "パウエル議長の発言",
      "BOJ Gov Ueda Speaks": "植田総裁の発言"}


def _title_ja(t: str) -> str:
    suffix = ""
    for en, ja in ((" m/m", " (前月比)"), (" y/y", " (前年比)"), (" q/q", " (前期比)")):
        if t.endswith(en):
            t, suffix = t[: -len(en)], ja
    if t in JA:
        return JA[t] + suffix
    for pre, ja in (("Prelim ", " 速報"), ("Advance ", " 速報"), ("Final ", " 確報"), ("Revised ", " 改定"), ("Flash ", " 速報")):
        if t.startswith(pre) and t[len(pre):] in JA:
            return JA[t[len(pre):]] + ja + suffix
    return t + suffix


def fetch() -> list[dict]:
    try:
        if os.path.exists(CACHE) and time.time() - os.path.getmtime(CACHE) < 3600:
            return json.load(open(CACHE))
        data = json.load(urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=15))
        os.makedirs(os.path.dirname(CACHE) or ".", exist_ok=True)
        json.dump(data, open(CACHE, "w"))
        return data
    except Exception:
        return json.load(open(CACHE)) if os.path.exists(CACHE) else []


def events(impacts=("High", "Medium"), currencies=("USD", "JPY")) -> list[dict]:
    out = []
    for x in fetch():
        if x.get("country") in currencies and x.get("impact") in impacts:
            t = pd.Timestamp(x["date"]).tz_convert("UTC")
            out.append({"time": t, "title": _title_ja(x["title"]), "title_en": x["title"], "impact": x["impact"],
                        "emoji": EMOJI.get(x["impact"], ""), "country": x["country"],
                        "forecast": x.get("forecast") or "", "previous": x.get("previous") or ""})
    return sorted(out, key=lambda e: e["time"])


def fmt(e: dict, with_values=True) -> str:
    s = f"{e['emoji']} {e['time'].tz_convert('Asia/Tokyo'):%m/%d %H:%M} {'米' if e['country'] == 'USD' else '日'}{e['title']}"
    if with_values and (e["forecast"] or e["previous"]):
        s += f" (予想 {e['forecast'] or '-'} / 前回 {e['previous'] or '-'})"
    return s


def upcoming(now: pd.Timestamp, hours: float, impacts=("High", "Medium")) -> list[dict]:
    return [e for e in events(impacts) if pd.Timedelta(0) <= e["time"] - now <= pd.Timedelta(hours=hours)]


def near(t: pd.Timestamp, minutes: int = 20, impacts=("High", "Medium")) -> list[dict]:
    """時刻 t の前後 minutes 分に発表された指標 (急変の原因の候補)"""
    return [e for e in events(impacts) if abs((e["time"] - t).total_seconds()) <= minutes * 60]
