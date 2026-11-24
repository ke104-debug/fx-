"""通知した売買を「保有中」として記録し、決済のタイミング (利確・損切り・期限・建値への移動) を通知する。
保有中の一覧は journal/positions.json (git で管理し、毎時の分析と5分監視の両方から使う)。
決済の通知は、活動時間にかかわらず送る (お金に直結するため)。"""
import json
import os
import time

PATH = os.path.join(os.path.dirname(__file__), "..", "journal", "positions.json")
CLOSED = os.path.join(os.path.dirname(__file__), "..", "journal", "closed.csv")
HOLD_H = {"H3": 14, "H4": 48, "H5": 24, "S": 72}
# 決済・ポジション管理部隊の出口の決め方 (reports/exits.md, 23年): 急変の反動 (S) と東京レンジ抜け (H3) は
# 利確を置かずに伸ばす方が作る期間・確かめる期間とも良かった → 1R 進んだら損切りを「最高値から 2ATR」に引き上げ続け、期限は3倍
TRAIL = {"S": dict(after_r=1.0, atr=2.0, hold_x=3), "H3": dict(after_r=1.0, atr=2.0, hold_x=3)}


def load() -> list[dict]:
    return json.load(open(PATH)) if os.path.exists(PATH) else []


def save(ps: list[dict]):
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    json.dump(ps, open(PATH, "w"), ensure_ascii=False, indent=1)


def conflict(side: int) -> str | None:
    """保有中との関係: 同じ向きを保有中なら "same"、逆向きを保有中なら "opposite" """
    ps = load()
    if any(p["side"] == side for p in ps):
        return "same"
    if any(p["side"] == -side for p in ps):
        return "opposite"
    return None


def weekend_note(now_epoch: int) -> tuple[str, str, int] | None:
    """金曜の夜 (日本時間 23時台) に保有中があれば、週末をまたぐ前の確認を送る"""
    t = time.gmtime(now_epoch + 9 * 3600)   # 日本時間
    if not (t.tm_wday == 4 and t.tm_hour == 23):
        return None
    ps = load()
    if not ps:
        return None
    lines = [f"・{'買い' if p['side'] == 1 else '売り'} {p['entry']:.2f}（損切り {p['stop']:.2f} / 利確 {p['target']:.2f}）" for p in ps]
    return ("📅 週末前の確認 USD/JPY",
            "保有中:\n" + "\n".join(lines) + "\n━━━━━━━━━━\n週明けは窓を開けて、損切りより不利な価格で決済されることがあります。\n"
            "週末をまたがずに決済するか、損切りの位置を見直すことをおすすめします。", 16776960)


def add(side: int, entry: float, stop: float, target: float, key: str, t_epoch: int, atr: float | None = None):
    ps = load()
    tr = TRAIL.get(key) if atr else None
    ps.append({"id": f"{key}-{t_epoch}", "side": side, "entry": round(entry, 3), "stop": round(stop, 3), "target": round(target, 3),
               "key": key, "opened": t_epoch, "expire": t_epoch + HOLD_H.get(key, 48) * 3600 * (tr["hold_x"] if tr else 1),
               "half_sent": False, **({"mode": "trail", "atr": round(atr, 4), "best": round(entry, 3), "risk": abs(entry - stop)} if tr else {})})
    save(ps)


def _record(p, px, why, t):
    import csv
    new = not os.path.exists(CLOSED)
    with open(CLOSED, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["id", "side", "entry", "exit", "pips", "result", "opened", "closed"])
        w.writerow([p["id"], p["side"], p["entry"], round(px, 3), round(p["side"] * (px - p["entry"]) * 100, 1), why,
                    time.strftime("%Y-%m-%d %H:%M", time.gmtime(p["opened"])), time.strftime("%Y-%m-%d %H:%M", time.gmtime(t))])


def check(bars: list[tuple], save_state: bool = True) -> list[tuple[str, str, int]]:
    """1分足 (t, o, h, l, c) で保有中を確認し、決済などの通知 (タイトル, 本文, 色) を返す"""
    ps, keep, out = load(), [], []
    now = bars[-1][0]
    for p in ps:
        side, e, stop, tgt = p["side"], p["entry"], p["stop"], p["target"]
        word = "買い" if side == 1 else "売り"
        done = None
        trail = p.get("mode") == "trail"
        moved = None
        for t, o, h, l, c in bars:
            if t < p.get("checked", p["opened"]):
                continue
            # 損切りを引き上げた足は、引き上げる前の損切りでは決済されていないので、もう一度は判定しない
            if t > p.get("moved_t", -1) and ((side == 1 and l <= stop) or (side == -1 and h >= stop)):
                done = (stop, "損切り" if side * (stop - e) < 0 else "追いかけの決済", t); break
            if not trail and ((side == 1 and h >= tgt) or (side == -1 and l <= tgt)):
                done = (tgt, "利確", t); break
            if trail:
                p["best"] = max(p["best"], h) if side == 1 else min(p["best"], l)
                cfg = TRAIL[p["key"]]
                if side * (p["best"] - e) >= cfg["after_r"] * p["risk"]:
                    ns = round(p["best"] - side * cfg["atr"] * p["atr"], 3)
                    ns = max(ns, e) if side == 1 else min(ns, e)          # 1R 進んだら少なくとも建値
                    if side * (ns - stop) >= 0.5 * p["atr"] or (side * (stop - e) < 0 <= side * (ns - e)):
                        stop = ns; p["stop"] = ns; moved = ns; p["moved_t"] = t
        c = bars[-1][4]
        if not done and now >= p["expire"]:
            done = (c, "期限で決済", now)
        if done:
            px, why, t = done
            pips = side * (px - e) * 100
            mark = {"利確": "✅", "損切り": "🛑", "期限で決済": "⏱", "追いかけの決済": "✅"}[why]
            title = f"{mark} {why} USD/JPY ({word} {e:.2f})"
            body = (f"**{why}: {px:.2f}**（{pips:+.0f} pips）\n"
                    f"エントリー {e:.2f} → 決済 {px:.2f}\n"
                    + ("損切りラインに到達しました。ルールどおり決済してください。" if why == "損切り" else
                       "引き上げた損切り (利益を守るライン) に到達しました。利益を確定してください。" if why == "追いかけの決済" else
                       "利確ラインに到達しました。" if why == "利確" else
                       "保有期限です。損益にかかわらず決済を推奨します。")
                    + "\n※注文で損切り・利確を入れていれば自動で決済済みです")
            out.append((title, body, 5763719 if why in ("利確", "追いかけの決済") else 15548997 if why == "損切り" else 9807270))
            if save_state:
                _record(p, px, why, t)
            continue
        if moved is not None:
            out.append((f"🔼 損切りの引き上げ USD/JPY ({word} {e:.2f})",
                        f"**損切りを {moved:.2f} へ**（現在 {c:.2f}、{side * (c - e) * 100:+.0f} pips）\n"
                        f"利確は置かずに伸ばす取引です。最も伸びた値から {TRAIL[p['key']]['atr']:.0f}ATR 後ろに損切りを付いていかせます。\n"
                        f"この損切りで決済されても {side * (moved - e) * 100:+.0f} pips です。", 3447003))
        # 利確までの半分まで進んだら、損切りを建値に動かす提案 (1回だけ。伸ばす取引では出さない)
        prog = side * (c - e) / max(abs(tgt - e), 1e-9)
        if not trail and prog >= 0.5 and not p.get("half_sent"):
            out.append((f"🔵 含み益 USD/JPY ({word} {e:.2f})",
                        f"**現在 {c:.2f}**（{side * (c - e) * 100:+.0f} pips、利確まで{(1 - prog) * 100:.0f}%）\n"
                        f"損切りを建値 {e:.2f} に移すと、負けなしで利確を待てます。", 3447003))
            p["half_sent"] = True
        p["checked"] = now
        keep.append(p)
    if save_state and (ps or os.path.exists(PATH)):
        save(keep)
    return out


def summary() -> str:
    ps = load()
    if not ps:
        return "保有中なし"
    return " / ".join(f"{'買' if p['side'] == 1 else '売'} {p['entry']:.2f}(損切 {p['stop']:.2f} 利確 {p['target']:.2f})" for p in ps)
