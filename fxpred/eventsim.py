"""重要指標 (🔴) の実況シミュレーション (お金は動かさない)。
発表の前に決めた作戦で、発表後の1分足を使って紙の上で売買し、結果を Discord に1回だけ送る。
  A 順張り5分 : 発表から5分の動きが普段の5分の1.5倍以上なら、その向きに入る。損切りは発表後5分の反対側の端の外、利確は損切り幅の2倍
  B 逆張り15分: 発表から15分の動きが普段の15分の1.5倍以上なら、逆向きに入る。損切り・利確は A と同じ考え方
  C 組織の判断: 発表前の最後の会議の結論が「買い/売りの合図」ならその向き、「様子見」なら入らない
  どれも発表の60分後に手仕舞い。往復0.8pipsを引く。
発表の65分後以降の5分監視で、1分足の記録から再現して計算する (途中の状態を持たないので、実行が遅れても結果は同じ)。
予想の通知 (発表の約1時間前) は毎時の分析 (scan.py) から forecast() で送る。記録: journal/eventsim.csv"""
import os
import statistics as stt

import pandas as pd

PATH = os.path.join(os.path.dirname(__file__), "..", "journal", "eventsim.csv")
ORG = os.path.join(os.path.dirname(__file__), "..", "journal", "org.csv")
COST = 0.008


def _events():
    from . import calendar as cal
    out = {}
    for e in cal.events(impacts=("High",)):
        out.setdefault(int(e["time"].timestamp()), []).append(e["title"])
    return out


def _done() -> set:
    try:
        return set(pd.read_csv(PATH).release_utc.astype(str))
    except Exception:
        return set()


def _org_before(r: int) -> str:
    try:
        d = pd.read_csv(ORG)
        d["t"] = pd.to_datetime(d.time_utc, utc=True)
        d = d[d.t <= pd.Timestamp(r, unit="s", tz="UTC")]
        return str(d.conclusion.iloc[-1]) if len(d) and "conclusion" in d else "記録なし"
    except Exception:
        return "記録なし"


def _manage(bars, i0, side, entry, stop, tgt, t_end):
    for t, o, h, l, c in bars[i0 + 1:]:
        if t >= t_end:
            break
        if (side == 1 and l <= stop) or (side == -1 and h >= stop):
            return stop, "損切り"
        if (side == 1 and h >= tgt) or (side == -1 and l <= tgt):
            return tgt, "利確"
    last = [b for b in bars if b[0] < t_end][-1]
    return last[4], "60分で手仕舞い"


def simulate(bars, r: int) -> dict | None:
    pre = [b for b in bars if b[0] < r]
    post = [b for b in bars if r <= b[0] < r + 3600]
    if len(pre) < 500 or len(post) < 50:
        return None
    p0 = pre[-1][4]
    closes = [b[4] for b in pre[-3000:]]
    sd5 = stt.median(abs(closes[i] - closes[i - 5]) for i in range(5, len(closes)))
    sd15 = stt.median(abs(closes[i] - closes[i - 15]) for i in range(15, len(closes)))
    res = {"release_utc": f"{pd.Timestamp(r, unit='s', tz='UTC'):%Y-%m-%d %H:%M}", "price_before": round(p0, 3),
           "high_60": round(max(b[2] for b in post), 3), "low_60": round(min(b[3] for b in post), 3),
           "move_60": round((post[-1][4] - p0) * 100, 1)}
    for name, mins, fade in (("A", 5, False), ("B", 15, True)):
        w = [b for b in post if b[0] < r + mins * 60]
        mv = w[-1][4] - p0
        sd = sd5 if mins == 5 else sd15
        res[f"move_{mins}"] = round(mv * 100, 1)
        if abs(mv) < 1.5 * sd or mv == 0:
            res[name] = ""
            continue
        side = (1 if mv > 0 else -1) * (-1 if fade else 1)
        entry = w[-1][4]
        stop = (min(b[3] for b in w) - 0.02) if side == 1 else (max(b[2] for b in w) + 0.02)
        tgt = entry + side * 2 * abs(entry - stop)
        i0 = bars.index(w[-1])
        px, why = _manage(bars, i0, side, entry, stop, tgt, r + 3600)
        res[name] = round((side * (px - entry) - COST) * 100, 1)
        res[name + "_why"] = f"{'買い' if side == 1 else '売り'} {entry:.3f}→{px:.3f} {why}"
    concl = _org_before(r)
    res["org"] = concl
    side = 1 if ("買い" in concl and "合図" in concl) else -1 if ("売り" in concl and "合図" in concl) else 0
    if side:
        px, why = _manage(bars, bars.index(post[0]) - 1, side, p0, p0 - side * 0.3, p0 + side * 0.6, r + 3600)
        res["C"] = round((side * (px - p0) - COST) * 100, 1)
    else:
        res["C"] = 0.0
    return res


def run(bars, now: int) -> list[tuple[str, str, int]]:
    """5分監視から呼ぶ。結果が出た指標があれば通知を返す"""
    done, out = _done(), []
    for r, titles in _events().items():
        key = f"{pd.Timestamp(r, unit='s', tz='UTC'):%Y-%m-%d %H:%M}"
        if key in done or not (r + 65 * 60 <= now <= r + 6 * 3600):
            continue
        res = simulate(bars, r)
        if not res:
            continue
        res["events"] = " / ".join(titles)[:200]
        old = pd.read_csv(PATH) if os.path.exists(PATH) else pd.DataFrame()
        allr = pd.concat([old, pd.DataFrame([res])])
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        allr.to_csv(PATH, index=False)
        f = lambda k: "入らず" if res.get(k, "") == "" else f"{res[k]:+.1f}pips ({res.get(k + '_why', '')})"
        tot = {k: pd.to_numeric(allr[k], errors="coerce").fillna(0).sum() for k in ("A", "B", "C")}
        body = (f"{res['events']}\n発表前 {res['price_before']:.3f} → 5分 {res['move_5']:+.1f} / 15分 {res['move_15']:+.1f} / "
                f"60分 {res['move_60']:+.1f}pips (高値 {res['high_60']:.3f} 安値 {res['low_60']:.3f})\n━━━━━━━━━━\n"
                f"A 順張り5分: {f('A')}\nB 逆張り15分: {f('B')}\nC 組織の判断「{res['org'][:30]}」: {res['C']:+.1f}pips\n━━━━━━━━━━\n"
                f"これまで{len(allr)}回の合計: A {tot['A']:+.1f} / B {tot['B']:+.1f} / C {tot['C']:+.1f} pips (紙の上の売買)")
        out.append((f"🧪 指標の実況シミュレーション結果 ({(pd.Timestamp(r, unit='s', tz='UTC') + pd.Timedelta(hours=9)):%m/%d %H:%M})", body, 9807270))
    return out


def forecast(h, res: dict, now=None) -> tuple[str, str] | None:
    """毎時の分析から: 30〜90分後に重要指標があれば、予想と作戦を1回だけ知らせる"""
    now = now or pd.Timestamp.now(tz="UTC")
    soon = [(r, t) for r, t in _events().items() if 30 * 60 <= r - now.timestamp() <= 90 * 60]
    if not soon:
        return None
    r, titles = soon[0]
    try:
        past = pd.read_csv(PATH)
        rng = ((past.high_60 - past.low_60) * 100).tail(20)
        exp = f"過去{len(rng)}回の発表後1時間の値幅の中央値 {rng.median():.0f}pips" if len(rng) >= 3 else None
    except Exception:
        exp = None
    exp = exp or f"今の1時間の平均的な値幅 {h.atr.iat[-1] * 100:.0f}pips の1.5倍前後 (記録がたまるまでの目安)"
    body = (f"{' / '.join(titles)[:200]}\n━━━━━━━━━━\n動きの大きさの目安: {exp}\n"
            f"全統括AI: {res.get('conclusion', '')}\n方向の会議: {res.get('verdict', '')} ({res.get('overall', 0):+.2f}) / リスク: {res.get('risk', {}).get('level', '')}\n"
            "━━━━━━━━━━\n発表後に紙の上で試す作戦: A 5分の動きに順張り / B 15分の動きに逆張り / C 組織の判断。結果は発表の約1時間後に送ります")
    return f"🧪 重要指標の予想 ({(pd.Timestamp(r, unit='s', tz='UTC') + pd.Timedelta(hours=9)):%m/%d %H:%M})", body
