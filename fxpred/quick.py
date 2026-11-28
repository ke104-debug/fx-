"""数分ごとの簡易監視 (軽量: Python 標準機能だけで動く。pandas 等のインストール不要)

1分足 (ほぼリアルタイム) を見て、次の時だけ動く:
  急変    : 5/15/30/60分の動きが普段の3.5倍以上、または60分で1円以上 → すぐ Discord に通知 (時間外でも)
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
    # 急変の通知の基準: 普段の3.5倍以上 (2026-11-26 に4倍から変更。15分で3.9倍の急落を通知できなかったため)
    shock = worst[1][1] >= 3.5 or abs(moves.get(60, (0, 0))[0]) >= 1.0
    return c, sd, moves, worst, shock


def to5m(bars, n=4):
    """1分足から、確定済みの5分足を直近 n 本作る (時刻が5分の倍数で区切る)"""
    groups = {}
    for t, o, h, l, c in bars[-(n + 1) * 5 - 5:]:
        groups.setdefault(t // 300, []).append((o, h, l, c))
    keys = sorted(groups)[:-1]          # 最後の (まだ確定していない) 5分足は使わない
    return [(groups[k][0][0], max(x[1] for x in groups[k]), min(x[2] for x in groups[k]), groups[k][-1][3]) for k in keys][-n:]


def confirmed(bars, side, lvl):
    """下値 (上値) に触れた後、5分足が陽線 (陰線) で戻したか。検証 (直近12週) で、すぐ入るより成績が良かった入り方"""
    b = to5m(bars, 3)
    if len(b) < 3:
        return False
    touched = min(x[2] for x in b) <= lvl + 0.05 if side == 1 else max(x[1] for x in b) >= lvl - 0.05
    o, h, l, c = b[-1]
    turned = (c > o and c > b[-2][3]) if side == 1 else (c < o and c < b[-2][3])
    return touched and turned


ASSETS = {"ES=F": "米国株先物 (S&P500)", "NIY=F": "日経平均先物", "ZN=F": "米国債先物 (10年)"}


def _series(ticker):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1m&range=5d"
    d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=15))
    r = d["chart"]["result"][0]
    return [(t, c) for t, c in zip(r["timestamp"], r["indicators"]["quote"][0]["close"]) if c is not None]


def cross_asset(start_epoch: int, direction: int) -> str | None:
    """急変と同じ時間に、株・日経・米国債がどう動いたか (Yahoo の先物は約10分遅れ)。
    ニュースを待たずに、リスクオン / リスクオフ / 米金利 / 円独自の要因 を見分ける"""
    z = {}
    for tk in ASSETS:
        try:
            ser = _series(tk)
        except Exception:
            continue
        if len(ser) < 300 or ser[-1][0] < start_epoch:
            continue
        before = [c for t, c in ser if t <= start_epoch]
        if not before:
            continue
        mv = ser[-1][1] - before[-1]
        mins = max((ser[-1][0] - start_epoch) / 60, 5)
        ch = [ser[i][1] - ser[i - 5][1] for i in range(5, len(ser), 5)]
        m = sum(ch) / len(ch); sd = math.sqrt(sum((x - m) ** 2 for x in ch) / len(ch)) or 1e-9
        z[tk] = mv / (sd * math.sqrt(mins / 5))
    if not z:
        return None
    stock = (z.get("ES=F", 0) + z.get("NIY=F", 0)) / (2 if "NIY=F" in z and "ES=F" in z else 1)
    bond = z.get("ZN=F", 0)   # 債券先物が下がる = 金利が上がる
    parts = []
    for tk, v in z.items():
        word = "上昇" if v > 1.5 else "下落" if v < -1.5 else "ほぼ横ばい"
        parts.append(f"{ASSETS[tk]} {word} ({v:+.1f})")
    if direction > 0 and stock > 1.5 and bond < -1:
        why = "リスクオン (株高・米金利上昇) による円売り・ドル買い"
    elif direction > 0 and stock > 1.5:
        why = "リスクオン (株高) による円売り"
    elif direction > 0 and bond < -1.5:
        why = "米金利の上昇によるドル買い"
    elif direction < 0 and stock < -1.5 and bond > 1:
        why = "リスクオフ (株安・米金利低下) による円買い"
    elif direction < 0 and stock < -1.5:
        why = "リスクオフ (株安) による円買い"
    elif direction < 0 and bond > 1.5:
        why = "米金利の低下によるドル売り"
    elif abs(stock) < 1 and abs(bond) < 1:
        why = "株・金利は静か → 円独自の要因 (介入・日銀・要人発言など) の可能性"
    else:
        why = "はっきりしない"
    return f"市場の連動 (約10分遅れ): {why}\n　" + " / ".join(parts)


def find_cause(start_epoch: int, direction: int) -> list[dict]:
    try:
        import pandas as pd
        from . import specialists as sp
        return sp.cause_of_move(sp.headlines(hours=3), pd.Timestamp(start_epoch, unit="s", tz="UTC"), direction)
    except Exception as e:
        print("ニュース取得エラー:", e)
        return []


def cause_text(causes: list[dict], start_epoch: int | None = None, direction: int = 0) -> str:
    ev_lines = []
    xa = None
    if start_epoch and direction:
        try:
            xa = cross_asset(start_epoch, direction)
        except Exception:
            xa = None
    if start_epoch:
        try:
            import pandas as pd
            from . import calendar as cal
            ev_lines = [cal.fmt(e) for e in cal.near(pd.Timestamp(start_epoch, unit="s", tz="UTC"), 20)]
        except Exception:
            pass
    head = (["同じ時刻に発表された指標 (Forex Factory)"] + ["・" + x for x in ev_lines] + [""]) if ev_lines else []
    if xa:
        head = [xa, ""] + head
    if not causes:
        return "\n".join(head + ["原因: まだ関連ニュースが見つかりません (見つかれば続報を送ります)"])
    lines = head + ["考えられる原因 (ニュースの見出しから推定)"]
    for x in causes:
        lines.append(f"・{x['time'].tz_convert('Asia/Tokyo'):%H:%M} {x['title'][:80]}")
    cats = sorted({c for x in causes for c in x["cats"]})
    if cats:
        lines.append("分類: " + "、".join(cats))
    lines.append("※速報・観測報道は後で否定されることがあります。公式発表で確認を")
    return "\n".join(lines)


def _strength_line():
    try:
        from .org import strength
        return "\n" + strength.line()
    except Exception:
        return ""


def run(dry=False):
    from . import notify
    t0 = time.time()
    bars = fetch_1m()
    if time.time() - bars[-1][0] > 15 * 60:
        print("市場が閉まっているか、データが止まっています"); return
    c, sd, moves, worst, shock = check(bars)
    from . import positions
    exits = positions.check(bars, save_state=not dry)
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    lv = json.load(open(LEVELS)) if os.path.exists(LEVELS) else {}
    now = int(time.time())
    out = list(exits)
    wk = positions.weekend_note(now)
    if wk and st.get("weekend_warned") != time.strftime("%Y-%m-%d", time.gmtime(now)):
        out.append(wk); st["weekend_warned"] = time.strftime("%Y-%m-%d", time.gmtime(now))
    # 0) 5分監視の補佐 (Groq): 今の状況・30分後の見立て・注意を記録 (急変の通知にも添える)。判断は変えない
    qa = None
    if not dry and not __import__("fxpred.org.mode", fromlist=["x"]).paused("5分監視の見立て"):
        try:
            from .org import quick_ai
            qa = quick_ai.read(bars, lv, st)
        except Exception as e:
            print("5分監視の補佐のエラー:", e)
    # 1) 急変 (時間外でも送る)。同じ向きの急変は30分に1回まで (さらに大きくなった時は再通知)
    if shock:
        k, (mv, z) = worst
        direction = "急騰" if mv > 0 else "急落"
        last = st.get("shock", {})
        if now - last.get("t", 0) > 1800 or last.get("dir") != direction or z >= last.get("z", 0) * 1.5:
            title = f"⚠️ {direction} USD/JPY {c:.2f}"
            causes = find_cause(now - k * 60, 1 if mv > 0 else -1)
            body = (f"**{k}分で {mv:+.2f}円**（普段の{z:.1f}倍）\n"
                    f"60分: {moves.get(60, (0, 0))[0]:+.2f}円 / 15分: {moves.get(15, (0, 0))[0]:+.2f}円\n"
                    "━━━━━━━━━━\n" + cause_text(causes, now - k * 60, 1 if mv > 0 else -1) +
                    + _strength_line()
                    + (f"\n【AIの見立て (Groq)】{qa['state']}" + (f" / 注意: {qa['caution']}" if qa.get("caution") else "") if qa else "")
                    + "\n━━━━━━━━━━\n今は入らないでください。\n下げ止まり・上げ止まりをAIが確認したら、売買の判断を通知します。")
            if not causes:
                st["cause_pending"] = {"start": now - k * 60, "dir": 1 if mv > 0 else -1, "until": now + 1800, "label": direction}
            out.append((title, body, 16776960))
            st["shock"] = {"t": now, "dir": direction, "z": z}
            st["pending_rebound"] = {"t": now, "dir": direction, "extreme": c}
    # 1b) 原因が見つかっていない急変は、30分以内に「原因の続報」を送る
    cp = st.get("cause_pending")
    if cp and not shock:
        if now > cp["until"]:
            st.pop("cause_pending", None)
        else:
            causes = find_cause(cp["start"], cp["dir"])
            if causes:
                out.append((f"📰 {cp['label']}の原因 (続報)", cause_text(causes, cp["start"], cp["dir"]), 3447003))
                st.pop("cause_pending", None)
    # 2) チャンスの候補 → AI の詳しい分析へ
    trigger = None
    if lv:
        sup, res = lv.get("support"), lv.get("resistance")
        # 下値・上値に近づき、さらに5分足で反発 (反落) を確認できた時だけ AI の詳しい分析へ
        if sup and abs(c - sup) <= 0.10:
            trigger = ("near_support", sup) if confirmed(bars, 1, sup) else ("wait", sup)
        elif res and abs(c - res) <= 0.10:
            trigger = ("near_resistance", res) if confirmed(bars, -1, res) else ("wait", res)
    pr = st.get("pending_rebound")
    if pr and now - pr["t"] < 6 * 3600:
        last5 = bars[-5:]
        turned = (last5[-1][4] > last5[0][1]) if pr["dir"] == "急落" else (last5[-1][4] < last5[0][1])
        moved_on = (c - pr["extreme"]) * (1 if pr["dir"] == "急落" else -1)
        if turned and moved_on > 0.05:
            trigger = ("rebound", pr["extreme"])
    if trigger and trigger[0] == "wait":
        print(f"下値・上値 {trigger[1]:.2f} 付近。5分足での反発確認待ち")
    elif trigger and now - st.get("chance_t", {}).get(trigger[0], 0) > 3600:
        st.setdefault("chance_t", {})[trigger[0]] = now
        try:
            from . import scan
            msg = scan.quick_chance(trigger, c)
            if msg:
                out.append(msg[:3])
                if len(msg) > 3 and msg[3]:
                    positions.add(*msg[3], t_epoch=now)
                if trigger[0] == "rebound":
                    st.pop("pending_rebound", None)
        except Exception as e:  # 詳しい分析に失敗しても、簡易監視は止めない
            print("AI分析エラー:", e)
    try:   # 速報の見張り役: 要人発言・介入・中央銀行などの新しい見出しを AI に読ませ、重要なものはすぐ通知 (時間外も)
        from . import newswatch
        if not dry:
            out += newswatch.check(st, c, now)
    except Exception as e:
        print("速報の見張りのエラー:", e)
    try:   # Discord でボスAIと話す (利用者のコメントにボスAIが返信)
        from .org import discord_chat
        discord_chat.check(st, dry=dry)
    except Exception as e:
        print("Discord の会話のエラー:", e)
    try:   # 流れ × タイミング (大きな流れ → 流れの変化 → 細かい流れ → トレンド転換ライン)。15分足の確定ごと
        from . import live_tl
        out += live_tl.check(st, now, c, dry=dry)
    except Exception as e:
        print("流れ×タイミングのエラー:", e)
    try:   # 持ち中の危ない場面をすぐ知らせる (すぐ利確・決済か、持ち続けるかを23年の結果つきで)
        from . import exit_alert
        for m_ in exit_alert.check(st, now):
            out.append(m_)
    except Exception as e:
        print("持ち中の危ない場面の確認のエラー:", e)
    try:   # 通知する作戦「急変に付いていく」: 毎時の足の確定直後に確かめる (遅れるほど不利なので5分監視で)
        from . import live_sf
        m_ = live_sf.check(st, now, c)
        if m_:
            out.append(m_[:3])
            if not dry:
                positions.add(*m_[3], t_epoch=now)
    except Exception as e:
        print("急変に付いていくのエラー:", e)
    if not dry:   # 重要指標の前後15分と急変の15分後のレポート (利用者の要望)
        try:
            from . import event_report as er
            for fn in (er.pre, er.post, er.shock_followup):
                m_ = fn(bars, now, st)
                if m_:
                    out.append(m_)
        except Exception as e:
            print("レポートのエラー:", e)
    if not dry:   # 重要指標の実況シミュレーション (発表の65分後以降に1回だけ結果を送る)
        try:
            from . import eventsim
            out += eventsim.run(bars, now)
        except Exception as e:
            print("指標のシミュレーションのエラー:", e)
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
