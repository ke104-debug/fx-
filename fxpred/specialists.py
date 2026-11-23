"""得意分野の違う3体の担当AIと、まとめ役 (戦略選択AI)。すべて無料の情報と手元の計算で動く。

  急変担当   : 何が起きたか (無料ニュースの見出しから原因を判定)、動きの大きさ、下げ止まり/上げ止まりの確認
  レンジ担当 : 下値・上値 (何度も反発した価格帯)、レンジ内の位置、行き過ぎ (RSI・ボリンジャー)
  トレンド担当: トレンドの向き・強さ・続いている時間、勢いの衰え、転換のサイン
  まとめ役   : 相場状態に応じて主担当を決め、他の担当の意見と AI_FX の判定 (edge) を合わせて最終判断を出す
"""
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import timedelta

import numpy as np
import pandas as pd

CAUSES = {
    "為替介入・当局": ["介入", "財務省", "財務相", "財務官", "円買い", "レートチェック", "口先"],
    "日銀": ["日銀", "植田", "決定会合", "日本銀行"],
    "FRB・米金利": ["FRB", "パウエル", "FOMC", "米金利", "利下げ", "利上げ示唆", "米国債"],
    "経済指標": ["雇用統計", "CPI", "PCE", "GDP", "指標", "物価", "小売"],
    "リスク回避・株式": ["株安", "急落", "NASDAQ", "ナスダック", "VIX", "リスク回避", "リスクオフ"],
    "政治・地政学": ["関税", "選挙", "地政学", "首脳", "政権", "戦争", "制裁"],
}


def headlines(query="ドル円", hours=24) -> list[dict]:
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": f"{query} when:1d", "hl": "ja", "gl": "JP", "ceid": "JP:ja"})
    try:
        xml = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read()
    except Exception:
        return []
    out, now = [], pd.Timestamp.now(tz="UTC")
    for it in ET.fromstring(xml).iter("item"):
        t = pd.to_datetime(it.findtext("pubDate"), utc=True, errors="coerce")
        if t is pd.NaT or now - t > timedelta(hours=hours):
            continue
        title = re.sub(r"\s*-\s*[^-]+$", "", it.findtext("title") or "")
        out.append({"time": t, "title": title, "link": it.findtext("link")})
    return sorted(out, key=lambda x: x["time"], reverse=True)


def classify(items: list[dict]) -> list[tuple[str, int, str]]:
    """原因ごとの見出し数 (多い順) と代表的な見出し"""
    res = []
    for cause, words in CAUSES.items():
        hit = [x for x in items if any(w in x["title"] for w in words)]
        if hit:
            res.append((cause, len(hit), hit[0]["title"]))
    return sorted(res, key=lambda x: -x[1])


# ---------------------------------------------------------------- 急変担当
def shock_view(h: pd.DataFrame, anom: dict, news: list[dict]) -> dict:
    causes = classify(news)
    n = len(h) - 1
    move24 = h.Close.iat[n] - h.Close.iat[n - 24]
    side = -1 if move24 < 0 else 1
    ext = h.Low.iloc[-24:].min() if side == -1 else h.High.iloc[-24:].max()
    stopped = (h.Low.iat[n] > h.Low.iat[n - 1] and h.Close.iat[n] > h.Open.iat[n]) if side == -1 else \
              (h.High.iat[n] < h.High.iat[n - 1] and h.Close.iat[n] < h.Open.iat[n])
    lines = [f"24時間の動き {move24:+.2f}円 (普段の{anom.get('z', 0):.1f}倍)"]
    if causes:
        lines.append("考えられる原因: " + "、".join(f"{c} ({k}件)" for c, k, _ in causes[:3]))
        lines.append(f"代表的な見出し: 「{causes[0][2][:60]}」")
    else:
        lines.append("原因につながる見出しは見つからず (原因不明の急変は撤退を優先)")
    lines.append(f"{'下げ' if side == -1 else '上げ'}止まりの確認: {'あり (直近の足)' if stopped else 'まだ'}")
    official = causes and causes[0][0] == "為替介入・当局"
    if official:
        lines.append("当局の介入は2〜5円動き、数日かけて一部を戻すことが多い。戻りは下げ幅の3〜4割を目安に")
    return {"name": "急変担当", "lines": lines, "extreme": round(ext, 2), "stopped": stopped, "causes": causes,
            "bias": 0 if not stopped else -side if causes else 0}


# ---------------------------------------------------------------- レンジ担当
def _swings(h, w=5, bars=120):
    x = h.iloc[-bars:]
    lows = x.Low[(x.Low == x.Low.rolling(2 * w + 1, center=True).min())]
    highs = x.High[(x.High == x.High.rolling(2 * w + 1, center=True).max())]
    return lows, highs


def _cluster(levels: pd.Series, tol=0.15) -> list[tuple[float, int]]:
    out = []
    for v in sorted(levels.values):
        if out and abs(v - out[-1][0]) <= tol:
            p, k = out[-1]; out[-1] = ((p * k + v) / (k + 1), k + 1)
        else:
            out.append((v, 1))
    return out


def range_view(h: pd.DataFrame, rsi: float) -> dict:
    c = h.Close.iat[-1]
    lows, highs = _swings(h)
    # 2回以上反発した価格帯を優先し、その中で今の価格に近いものを選ぶ (0.05円以内は「今いる場所」なので除く)
    sup = [lv for lv in _cluster(lows) if lv[0] < c - 0.05]
    res = [lv for lv in _cluster(highs) if lv[0] > c + 0.05]
    pick = lambda xs, near: (max if near == "max" else min)(
        [x for x in xs if x[1] >= 2] or xs, key=lambda x: x[0])
    s = pick(sup, "max") if sup else (h.Low.iloc[-120:].min(), 1)
    r = pick(res, "min") if res else (h.High.iloc[-120:].max(), 1)
    pos = (c - s[0]) / (r[0] - s[0]) if r[0] > s[0] else 0.5
    mid, sd = h.Close.rolling(80).mean().iat[-1], h.Close.rolling(80).std().iat[-1]
    z = (c - mid) / sd
    lines = [f"下値 {s[0]:.2f} ({s[1]}回反発) / 上値 {r[0]:.2f} ({r[1]}回反落) / レンジ内の位置 {pos:.0%}",
             f"RSI {rsi:.0f}、ボリンジャー {z:+.1f}σ"]
    bias = 0
    if pos <= 0.25 and rsi < 45:
        bias = 1; lines.append("下値に近く売られ気味 → 反発の買いを狙いやすい位置")
    elif pos >= 0.75 and rsi > 55:
        bias = -1; lines.append("上値に近く買われ気味 → 反落の売り、または買いの利確の位置")
    else:
        lines.append("レンジの中ほど → 端に寄るまで待つのが有利")
    return {"name": "レンジ担当", "lines": lines, "support": round(s[0], 2), "resistance": round(r[0], 2),
            "pos": pos, "bias": bias}


# ---------------------------------------------------------------- トレンド担当
def trend_view(h: pd.DataFrame) -> dict:
    c, ema50, ema200 = h.Close, h.Close.ewm(span=50).mean(), h.Close.ewm(span=200).mean()
    up = ema50 > ema200
    flips = np.flatnonzero(up.values[1:] != up.values[:-1])
    age = len(h) - 1 - (flips[-1] + 1) if len(flips) else len(h)
    # 過去のトレンドの長さ (中央値) と比べて、今のトレンドがどのくらい続いているか
    lengths = np.diff(flips) if len(flips) > 2 else np.array([age])
    med = float(np.median(lengths))
    direction = "上昇" if up.iat[-1] else "下降"
    atr = (h.High - h.Low).rolling(14).mean().iat[-1]
    stretch = (c.iat[-1] - ema50.iat[-1]) / atr
    lines = [f"{direction}トレンド (1時間足の50/200本平均) が{age}時間継続。過去の中央値は{med:.0f}時間"]
    warn = []
    if age > 1.5 * med:
        warn.append("通常より長く続いており、終盤の可能性")
    if abs(stretch) > 3:
        warn.append(f"平均線から{abs(stretch):.1f}本分離れ、行き過ぎ")
    rsi = h.rsi if "rsi" in h else None
    w = h.iloc[-48:]
    if rsi is not None:
        r = rsi.iloc[-48:]
        if direction == "上昇" and w.High.iat[-1] >= w.High.max() - 1e-9 and r.iat[-1] < r.max() - 5:
            warn.append("高値更新なのに RSI が下がる (勢いの衰え)")
        if direction == "下降" and w.Low.iat[-1] <= w.Low.min() + 1e-9 and r.iat[-1] > r.min() + 5:
            warn.append("安値更新なのに RSI が上がる (勢いの衰え)")
    lows, highs = _swings(h, bars=72)
    if direction == "上昇" and len(lows) and c.iat[-1] < lows.iloc[-1]:
        warn.append(f"直近の押し安値 {lows.iloc[-1]:.2f} を割り込み → 転換の初期サイン")
    if direction == "下降" and len(highs) and c.iat[-1] > highs.iloc[-1]:
        warn.append(f"直近の戻り高値 {highs.iloc[-1]:.2f} を上抜け → 転換の初期サイン")
    status = "転換警戒" if any("転換" in x for x in warn) else "減速" if warn else "継続"
    lines.append(f"判定: {status}" + (" (" + "、".join(warn) + ")" if warn else ""))
    bias = (1 if direction == "上昇" else -1) if status == "継続" else 0 if status == "減速" else (-1 if direction == "上昇" else 1)
    return {"name": "トレンド担当", "lines": lines, "direction": direction, "status": status, "age": age, "bias": bias}


# ---------------------------------------------------------------- まとめ役
LEAD = {"荒れ": "急変担当", "通常": "レンジ担当", "トレンド": "トレンド担当"}


def coordinate(regime: str, views: dict, learn: dict | None = None) -> dict:
    """主担当の意見を重く (2倍)、他の担当は1倍で合計。自己改善の成績で担当ごとの重みを調整する"""
    learn = learn or {}
    lead = LEAD[regime]
    score = 0.0
    for name, v in views.items():
        w = (2.0 if name == lead else 1.0) * learn.get("weights", {}).get(name, 1.0)
        score += w * v.get("bias", 0)
    verdict = "買い寄り" if score >= 1.5 else "売り寄り" if score <= -1.5 else "中立"
    return {"lead": lead, "score": round(score, 2), "verdict": verdict}
