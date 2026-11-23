"""数分ごとの簡易監視 (軽量: Python 標準機能だけで動く。pandas 等のインストール不要)

1分足 (ほぼリアルタイム) を見て、次の時だけ動く:
  急変    : 5/15/30/60分の動きが普段の4倍以上、または60分で1円以上 → すぐ Discord に通知 (時間外でも)
  チャンス: 毎時のAI分析が決めた下値・上値 (state/levels.json) に近づいた / 抜けた
            → AI の詳しい分析 (fxpred.scan.quick_chance) を呼び、信頼度が基準以上なら通知 (活動時間内)
普段は何も送らず、数秒で終わる。
"""
import json
import math
import os
import time
import urllib.request

STATE = os.getenv("FX_QUICK_STATE", "state/quick.json")
LEVELS = os.getenv("FX_LEVELS", "state/levels.json")


def fetch_1m():
    url = "https://query1.finance.yahoo.com/v8/finance/chart/USDJPY=X?interval=1m&range=5d"
    d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20))
    r = d["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    bars = [(t, o, h, l, c) for t, o, h, l, c in zip(r["timestamp"], q["open"], q["high"], q["low"], q["close"])
            if None not in (o, h, l, c)]
    return bars


def sigma5(bars):
    """普段の5分間の動き (標準偏差、直近5日)"""
    ch = [bars[i][4] - bars[i - 5][4] for i in range(5, len(bars), 5)]
    m = sum(ch) / len(ch)
    return math.sqrt(sum((x - m) ** 2 for x in ch) / len(ch))


def check(bars):
    c = bars[-1][4]
    sd = sigma5(bars)
    moves = {}
    for k in (5, 15, 30, 60):
        if len(bars) > k:
            mv = c - bars[-1 - k][4]
            moves[k] = (mv, abs(mv) / (sd * math.sqrt(k / 5)))
    worst = max(moves.items(), key=lambda kv: kv[1][1])
    shock = worst[1][1] >= 4 or abs(moves.get(60, (0, 0))[0]) >= 1.0
    return c, sd, moves, worst, shock


def run(dry=False):
    from . import notify
    t0 = time.time()
    bars = fetch_1m()
    if time.time() - bars[-1][0] > 15 * 60:
        print("市場が閉まっているか、データが止まっています"); return
    c, sd, moves, worst, shock = check(bars)
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    lv = json.load(open(LEVELS)) if os.path.exists(LEVELS) else {}
    now = int(time.time())
    out = []
    # 1) 急変 (時間外でも送る)。同じ向きの急変は30分に1回まで (さらに大きくなった時は再通知)
    if shock:
        k, (mv, z) = worst
        direction = "急騰" if mv > 0 else "急落"
        last = st.get("shock", {})
        if now - last.get("t", 0) > 1800 or last.get("dir") != direction or z >= last.get("z", 0) * 1.5:
            title = f"⚠️ {direction} USD/JPY {c:.2f}"
            body = (f"**{k}分で {mv:+.2f}円**（普段の{z:.1f}倍）\n"
                    f"60分: {moves.get(60, (0, 0))[0]:+.2f}円 / 15分: {moves.get(15, (0, 0))[0]:+.2f}円\n"
                    "━━━━━━━━━━\n今は入らないでください。\n下げ止まり・上げ止まりをAIが確認したら、売買の判断を通知します。")
            out.append((title, body, 16776960))
            st["shock"] = {"t": now, "dir": direction, "z": z}
            st["pending_rebound"] = {"t": now, "dir": direction, "extreme": c}
    # 2) チャンスの候補 → AI の詳しい分析へ
    trigger = None
    if lv:
        sup, res = lv.get("support"), lv.get("resistance")
        if sup and abs(c - sup) <= 0.10: trigger = ("near_support", sup)
        elif res and abs(c - res) <= 0.10: trigger = ("near_resistance", res)
    pr = st.get("pending_rebound")
    if pr and now - pr["t"] < 6 * 3600:
        last5 = bars[-5:]
        turned = (last5[-1][4] > last5[0][1]) if pr["dir"] == "急落" else (last5[-1][4] < last5[0][1])
        moved_on = (c - pr["extreme"]) * (1 if pr["dir"] == "急落" else -1)
        if turned and moved_on > 0.05:
            trigger = ("rebound", pr["extreme"])
    if trigger and now - st.get("chance_t", {}).get(trigger[0], 0) > 3600:
        st.setdefault("chance_t", {})[trigger[0]] = now
        try:
            from . import scan
            msg = scan.quick_chance(trigger, c)
            if msg:
                out.append(msg)
                if trigger[0] == "rebound":
                    st.pop("pending_rebound", None)
        except Exception as e:  # 詳しい分析に失敗しても、簡易監視は止めない
            print("AI分析エラー:", e)
    for title, body, color in out:
        print(title); print(body)
        if not dry:
            notify.send(title, body, color=color)
    st["last"] = {"t": now, "price": c, "z": round(worst[1][1], 1)}
    os.makedirs(os.path.dirname(STATE) or ".", exist_ok=True)
    json.dump(st, open(STATE, "w"), ensure_ascii=False, indent=1)
    print(f"{time.strftime('%H:%M', time.gmtime())} UTC {c:.3f} 最大の動き {worst[0]}分 {worst[1][0]:+.3f}円 (普段の{worst[1][1]:.1f}倍) "
          f"急変{'あり' if shock else 'なし'} チャンス候補 {trigger[0] if trigger else 'なし'} ({time.time() - t0:.1f}秒)")


if __name__ == "__main__":
    import sys
    run(dry="--dry" in sys.argv)
