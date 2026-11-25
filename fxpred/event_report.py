"""重要指標 (🔴) の前後15分と、大きな出来事 (急変) の時のレポート (利用者の要望)。5分監視から呼ぶ。
- 発表の約15分前: 今の価格・直近の動き・全統括AIの結論とリスク・指標前の傾向・調査班の報告・発表後の見方
- 発表の約15分後: 結果 (ニュースの見出し) と予想・5分/15分の動き (普段の何倍か)・高値/安値・AI の読み (Groq) と判断の目安
- 急変の約15分後: その後の動き (続いたか戻ったか)・原因らしいニュース・AI の読み
判断の目安は、検証済みのものと試しのものをはっきり分けて書く。ルールは変えない。"""
import statistics as stt

import pandas as pd

JST = pd.Timedelta(hours=9)


def _events():
    from . import calendar as cal
    out = {}
    for e in cal.events(impacts=("High",)):
        out.setdefault(int(e["time"].timestamp()), []).append(e)
    return out


def _sd(bars, k):
    cl = [b[4] for b in bars[-3000:]]
    return stt.median(abs(cl[i] - cl[i - k]) for i in range(k, len(cl))) or 1e-9


def _org():
    try:
        import os
        og = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "journal", "org.csv")).iloc[-1]
        return f"{og.get('conclusion', '')} / 方向の会議 {og.get('verdict', '')} {float(og.get('overall', 0)):+.2f} / リスク {og.get('risk', '')}"
    except Exception:
        return "記録なし"


def _heads(names, since, hours=1):
    from . import specialists as sp
    seen, out = set(), []
    for q in names:
        for x in sp.headlines(query=q, hours=hours):
            if x["time"] >= since and x["title"] not in seen:
                seen.add(x["title"]); out.append(x)
    return sorted(out, key=lambda x: x["time"])


def _ai(prompt):
    try:
        from .org import groq
        r, _ = groq.ask("指標の報告", prompt)
        return r if isinstance(r, dict) else {}
    except Exception:
        return {}


def pre(bars, now: int, st: dict):
    for r, evs in _events().items():
        key = f"pre-{r}"
        if not (r - 20 * 60 <= now <= r - 8 * 60) or st.get(key):
            continue
        st[key] = 1
        c = bars[-1][4]
        ch60 = (c - bars[-61][4]) * 100 if len(bars) > 61 else 0
        names = " / ".join(f"{e['title']} (予想 {e['forecast'] or '-'} / 前回 {e['previous'] or '-'})" for e in evs[:4])
        try:
            from .org import research
            rr = research.recent(hours=4, n=3)
        except Exception:
            rr = []
        body = (f"**{(pd.Timestamp(r, unit='s', tz='UTC') + JST):%H:%M} {names}**\n"
                f"今 {c:.3f} (1時間 {ch60:+.0f}pips)\n全統括AI: {_org()}\n━━━━━━━━━━\n"
                "・発表前に入るのは値が飛びやすく賭けになるので、システムは入らない\n"
                "・過去の傾向 (2022年以降の米雇用統計): 前日〜当日は買いの決済で下げやすい (試し・回数少)\n"
                "・発表直後の最初の動きは続かないことが多い (2022年以降、続いたのは37%)\n"
                "・発表から約15分後に、結果と値動きのレポートを送ります")
        if rr:
            body += "\n━━━━━━━━━━\n調査班:\n" + "\n".join(f"・{x[:70]}" for x in rr)
        return ("📣 重要指標の15分前", body, 9807270)
    return None


def post(bars, now: int, st: dict):
    for r, evs in _events().items():
        key = f"post-{r}"
        if not (r + 15 * 60 <= now <= r + 35 * 60) or st.get(key):
            continue
        st[key] = 1
        pre_ = [b for b in bars if b[0] < r]
        post_ = [b for b in bars if b[0] >= r]
        if not pre_ or not post_:
            return None
        p0 = pre_[-1][4]
        w5 = [b for b in post_ if b[0] < r + 300] or post_[:1]
        w15 = [b for b in post_ if b[0] < r + 900] or post_
        m5, m15 = (w5[-1][4] - p0) * 100, (w15[-1][4] - p0) * 100
        x5, x15 = abs(m5) / (_sd(pre_, 5) * 100), abs(m15) / (_sd(pre_, 15) * 100)
        hi, lo, now_px = max(b[2] for b in post_), min(b[3] for b in post_), post_[-1][4]
        since = pd.Timestamp(r, unit="s", tz="UTC")
        keys = list(dict.fromkeys([e["title"].split(" (")[0][:12] for e in evs] + ["ドル円"]))
        hs = _heads(keys, since)
        heads = "\n".join(f"・{(x['time'] + JST):%H:%M} {x['title'][:70]}" for x in hs[:6]) or "・結果の見出しがまだ見つからない"
        if x15 >= 1.5:
            guide = (f"15分で普段の{x15:.1f}倍動いた。過去の傾向では大きな最初の動きは戻りやすい → "
                     f"逆張り ({'売り' if m15 > 0 else '買い'}) が候補。ただし検証は数回分の試し。入るなら少額・損切りは15分の{'高値' if m15 > 0 else '安値'}の外")
        else:
            guide = f"15分の動きは普段の{x15:.1f}倍で小さい。方向が出ていないので見送りが基本"
        ai = _ai("FX (ドル円) の重要指標の発表後のレポート。以下だけを根拠に、結果と予想のずれ、市場の反応の読み方を短く。"
                 '推測で数字を作らない。JSON だけ: {"結果": "40字以内", "読み方": "50字以内", "判断": "見送り/買い候補/売り候補", "理由": "40字以内"}\n\n'
                 f"指標: {' / '.join(e['title'] + ' 予想 ' + (e['forecast'] or '-') for e in evs[:4])}\n見出し:\n{heads}\n"
                 f"値動き: 発表前 {p0:.3f} / 5分 {m5:+.1f}pips / 15分 {m15:+.1f}pips / 高値 {hi:.3f} 安値 {lo:.3f} / 今 {now_px:.3f}")
        body = (f"**{(since + JST):%H:%M} {' / '.join(e['title'] for e in evs[:3])}**\n{heads}\n━━━━━━━━━━\n"
                f"発表前 {p0:.3f} → 5分 {m5:+.1f}pips (普段の{x5:.1f}倍) → 15分 {m15:+.1f}pips (普段の{x15:.1f}倍)\n"
                f"発表後の高値 {hi:.3f} / 安値 {lo:.3f} / 今 {now_px:.3f}\n━━━━━━━━━━\n目安: {guide}")
        if ai:
            body += f"\nAIの読み (補助): {ai.get('読み方', '')} → {ai.get('判断', '')} ({ai.get('理由', '')})"
        body += "\n全統括AI: " + _org() + "\n※システムの作戦の条件がそろえば、別に売買の合図を送ります"
        return ("📣 重要指標の15分後", body, 9807270)
    return None


def shock_followup(bars, now: int, st: dict):
    """急変の通知から約15分後に、その後どうなったか"""
    sh = st.get("shock") or {}
    t = sh.get("t")
    if not t or st.get("shock_followed") == t or not (t + 15 * 60 <= now <= t + 40 * 60):
        return None
    st["shock_followed"] = t
    at = [b for b in bars if b[0] <= t]
    if not at:
        return None
    p0 = at[-1][4]
    after = [b for b in bars if b[0] > t]
    if not after:
        return None
    c = after[-1][4]
    hi, lo = max(b[2] for b in after), min(b[3] for b in after)
    up = sh.get("dir") == "急騰"
    cont = (c - p0) * (1 if up else -1) * 100
    hs = _heads(["ドル円", "為替 介入", "円相場"], pd.Timestamp(t - 1800, unit="s", tz="UTC"), hours=2)
    heads = "\n".join(f"・{(x['time'] + JST):%H:%M} {x['title'][:70]}" for x in hs[-5:]) or "・目立つ見出しなし"
    body = (f"{sh.get('dir', '急変')} の通知 ({(pd.Timestamp(t, unit='s', tz='UTC') + JST):%H:%M}, {p0:.3f}) から約15分\n"
            f"今 {c:.3f} (同じ向きに {cont:+.0f}pips) / その後の高値 {hi:.3f} 安値 {lo:.3f}\n"
            + ("→ 動きが続いている" if cont > 5 else "→ 戻している" if cont < -5 else "→ 止まっている")
            + f"\n━━━━━━━━━━\n{heads}\n全統括AI: {_org()}")
    return ("📣 急変の15分後", body, 9807270)
