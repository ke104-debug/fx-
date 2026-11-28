"""速報の見張り役 (利用者の要望 2026-11-27: ニュースや要人発言で状況は急変するので大切にしたい)。
5分監視から毎回呼ぶ。値が動く前に、要人発言・介入・中央銀行・米政権などの新しい見出しを見つけたら、
AI (Groq の無料枠、だめなら Gemini) に重要度とドル円への影響を読ませ、重要なものはすぐ通知する (時間外も)。
記録 journal/news_alerts.csv に、通知した時の値段を残し、夜の検証部屋で15分後・60分後の動きと答え合わせする (AI の読みが当たるかを採点)。
売買の合図は出さない (読みはまだ検証されていない)。"""
import os
import time

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
LOG = os.path.join(ROOT, "journal", "news_alerts.csv")
QUERIES = [("為替介入 OR 財務官 OR 財務相 為替 OR 円安けん制", "ja"), ("日銀 総裁 OR 植田 OR 利上げ OR 審議委員", "ja"),
           ("FRB OR パウエル OR FOMC OR 米利下げ", "ja"), ("ドル円 急", "ja"),
           ("Bank of Japan OR Ueda OR yen intervention", "en"), ("Fed Powell OR FOMC OR Treasury yields surge", "en")]
JST = pd.Timedelta(hours=9)


def check(st: dict, price: float, now_epoch: int, window_min: int = 40, weekend: bool = False) -> list[tuple[str, str, int]]:
    from . import specialists as sp
    seen = set(st.get("news_seen", []))
    now = pd.Timestamp(now_epoch, unit="s", tz="UTC")
    fresh = []
    for q, lang in QUERIES:
        for x in sp.headlines(query=q, hours=max(1, -(-window_min // 60)), lang=lang)[:10]:
            k = x["title"][:60]
            if k in seen:
                continue
            seen.add(k)
            if now - x["time"] <= pd.Timedelta(minutes=window_min):
                fresh.append(x)
    st["news_seen"] = list(seen)[-400:]
    if not fresh or not st.get("news_primed"):
        st["news_primed"] = True          # 初回は今ある見出しを覚えるだけ (古いものを速報にしない)
        return []
    lines = "\n".join(f"- {(x['time'] + JST):%m/%d %H:%M} {x['title']}" for x in fresh[:12])
    from .org import groq
    r, who = groq.ask("ニュース速報", "あなたはドル円の速報デスク。次の新しい見出しのうち、ドル円を数十pips以上動かしうるものだけを選ぶ。"
                      "見出しに書かれていることだけを使い、推測で数字を作らない。JSON だけ: "
                      '{"items": [{"no": 見出しの番号 (1から), "重要度": 0〜3 (3=介入・政策の驚き・要人の強い発言), "影響": -2〜2 (ドル円が上がる=+), "要点": "40字以内"}]}\n\n'
                      + "\n".join(f"{i + 1}. {x['title']}" for i, x in enumerate(fresh[:12])))
    items = r.get("items", []) if isinstance(r, dict) else []
    out, rows = [], []
    for it in items:
        try:
            imp, eff, no = int(it.get("重要度", 0)), int(it.get("影響", 0)), int(it.get("no", 0)) - 1
        except Exception:
            continue
        if imp < 2 or not (0 <= no < len(fresh)):
            continue
        x = fresh[no]
        rows.append({"time_utc": f"{now:%Y-%m-%d %H:%M}", "headline_utc": f"{x['time']:%Y-%m-%d %H:%M}", "title": x["title"][:120],
                     "importance": imp, "effect": eff, "summary": str(it.get("要点", ""))[:60], "price": round(price, 3), "ai": who})
        arrow = "↑ ドル高・円安の方向" if eff > 0 else "↓ ドル安・円高の方向" if eff < 0 else "向きは不明"
        out.append((f"📰 速報 (重要度{imp}) USD/JPY {price:.3f}",
                    f"**{it.get('要点', '')}**\n{(x['time'] + JST):%H:%M} {x['title'][:90]}\n━━━━━━━━━━\nAI の読み ({who}): {arrow} (強さ {eff:+d})\n"
                    "※AI の読みはまだ検証中 (答え合わせを記録しています)。売買の合図ではありません。急変すれば急変の通知と作戦の合図が別に届きます"
                    + ("\n🗓️ 市場は週末で閉まっています。月曜の開始 (日本時間 朝6〜7時) に窓を開けて始まる可能性があります。値段は金曜の終値" if weekend else ""),
                    15105570 if imp >= 3 else 16776960))
    if rows:
        old = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame()
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        pd.concat([old, pd.DataFrame(rows)]).to_csv(LOG, index=False)
    return out[:3]


def score(h1m: pd.DataFrame | None = None) -> list[str]:
    """夜の検証部屋: 速報の後15分・60分の動きが AI の読みの向きと合っていたか"""
    if not os.path.exists(LOG):
        return ["速報の見張り: まだ記録なし"]
    d = pd.read_csv(LOG)
    try:
        import yfinance as yf
        x = yf.download("USDJPY=X", period="7d", interval="1m", progress=False)
        x.columns = x.columns.get_level_values(0); x.index = pd.to_datetime(x.index, utc=True)
        c = x.Close
    except Exception:
        return ["速報の見張り: 値段を取れず採点できなかった"]
    res = []
    for r in d.itertuples():
        t = pd.Timestamp(r.time_utc, tz="UTC")
        if t + pd.Timedelta(minutes=60) > c.index[-1] or t < c.index[0]:
            continue
        p0 = c.asof(t); m15 = (c.asof(t + pd.Timedelta(minutes=15)) - p0) * 100; m60 = (c.asof(t + pd.Timedelta(minutes=60)) - p0) * 100
        if r.effect:
            res.append((r.importance, (m15 > 0) == (r.effect > 0), (m60 > 0) == (r.effect > 0), abs(m60)))
    if not res:
        return [f"速報の見張り: 記録 {len(d)}件 (答え合わせできる向きつきはまだなし)"]
    s = pd.DataFrame(res, columns=["imp", "h15", "h60", "mv"])
    return [f"速報の見張り: 向きつき {len(s)}件、AI の読みの当たり 15分後 {s.h15.mean():.0%} / 60分後 {s.h60.mean():.0%}、60分の動きの平均 {s.mv.mean():.0f}pips"]


def weekend_run(dry=False):
    """週末 (市場が閉まっている間) の速報の見張り (利用者の指示 2026-11-28: 土日もニュースを取得する。見逃すと月曜の相場に影響)。
    1時間ごとに news-watch.yml から呼ぶ。直近70分の新しい見出しだけを読む。状態は5分監視と同じ state/quick.json"""
    import json
    from . import notify, quick
    st = json.load(open(quick.STATE)) if os.path.exists(quick.STATE) else {}
    try:
        price = float(quick.fetch_1m()[-1][4])
    except Exception:
        price = float("nan")
    out = check(st, price, int(time.time()), window_min=70, weekend=True)
    for title, body, color in out:
        print(title); print(body)
        if not dry:
            notify.send(title, body, color=color)
    if not dry:
        os.makedirs(os.path.dirname(quick.STATE) or ".", exist_ok=True)
        json.dump(st, open(quick.STATE, "w"), ensure_ascii=False)
    print(f"週末の速報の見張り: 通知 {len(out)}件")


if __name__ == "__main__":
    import sys
    weekend_run(dry="--dry" in sys.argv)
