"""持ち中の危ない場面をすぐ知らせる (利用者の要望 2026-11-27: 途中でトレンドが変わり、すぐ利確しなければ損が見込まれる時はすぐ通知)。
5分監視が毎時の足の確定直後 (0〜12分) に、見張り中の取引ごとに確かめる。同じ場面は1回だけ。
すぐ降りるか・持ち続けるかの勧めは、23年の検証 (提案に記録) の結果どおり:
  流れ買い: 含み益1R以上の後に4時間足の向きが逆へ → すぐ利確 (35回: 持ち続けた +1.15R / すぐ降りた +1.41R、作る・確かめともに降りる方が得)
  急変の反動: 逆向きの急変 (3.5倍) で大きな流れも逆へ → すぐ決済 (5回: 持ち続けた -1.00R / すぐ -0.53R)
  逆向きの急変 (2.5倍) だけ → 知らせるが、持ち続ける (急変に付いていく 32回: 持ち続けた +0.05R / すぐ -0.22R、流れ買い 97回: +0.19R / 0.00R)"""
import numpy as np
import pandas as pd

STATS = {
    ("TF", "4h"): ("すぐ利確を推奨", "23年で同じ場面35回: 持ち続けた +1.15R / すぐ降りた +1.41R (降りた方が平均 +0.27R 得)"),
    ("S", "shock_trend"): ("すぐ決済を推奨", "23年で同じ場面5回: 5回とも損切りまで行った (持ち続けた -1.00R / すぐ -0.53R)"),
    ("SF", "shock"): ("損切りは置いたまま持ち続ける", "23年で同じ場面32回: 持ち続けた +0.05R / すぐ降りた -0.22R (急変は戻ることが多い)"),
    ("TF", "shock"): ("損切りは置いたまま持ち続ける", "23年で同じ場面97回: 持ち続けた +0.19R / すぐ降りた ±0R"),
    ("S", "shock"): ("損切りは置いたまま様子見", "23年で同じ場面33回: 持ち続けた +0.18R / すぐ +0.26R (差ははっきりしない)"),
}


def check(st: dict, now_epoch: int) -> list[tuple[str, str, int]]:
    now = pd.Timestamp(now_epoch, unit="s", tz="UTC")
    if now.minute > 12:
        return []
    slot = f"{now:%Y-%m-%d %H}"
    if st.get("exit_alert_slot") == slot:
        return []
    st["exit_alert_slot"] = slot
    from . import positions, scan, regime as rg
    ps = positions.load()
    mine = [p for p in ps if p.get("key") in ("SF", "S", "TF", "U")]
    if not mine:
        return []
    h = rg.prepare(scan.load_recent())
    C, Hh, L = h.Close.values, h.High.values, h.Low.values
    r = np.diff(C, prepend=C[0])
    sd = pd.Series(r).rolling(480, min_periods=200).std().values
    c4 = h.Close.resample("4h").last().dropna()
    up4 = (c4.ewm(span=20).mean() > c4.ewm(span=50).mean()).shift(1).reindex(h.index, method="ffill").fillna(False).values
    j = len(h) - 1
    out, changed = [], False
    for p in mine:
        side, e = p["side"], p["entry"]
        key = p["key"] if p["key"] != "U" else ("TF" if abs(e - p["stop"]) > 1.0 else "SF")   # 利用者の取引は近い作戦の結果で
        risk = p.get("risk0") or p.get("risk") or abs(e - p["stop"])
        t0 = pd.Timestamp(p["opened"], unit="s", tz="UTC")
        w = h[h.index >= t0]
        if w.empty or risk <= 0:
            continue
        best = w.High.max() if side == 1 else w.Low.min()
        mfe = side * (best - e)
        z = max(abs(C[j] - C[j - 1]) / sd[j], abs(C[j] - C[j - 3]) / (sd[j] * np.sqrt(3)))
        against = side * (C[j] - C[j - 3]) < 0
        trend_against = bool(h.trend_up.iat[j]) != (side == 1)
        flags = p.setdefault("alerts", [])
        found = []
        if key == "TF" and mfe >= risk and bool(up4[j]) != (side == 1) and "4h" not in flags:
            found.append("4h")
        if key == "S" and z >= 3.5 and against and trend_against and "shock_trend" not in flags:
            found.append("shock_trend")
        if z >= 2.5 and against and "shock" not in flags and not found:
            found.append("shock")
        for f in found:
            if (key, f) not in STATS:
                continue
            act, why = STATS[(key, f)]
            flags.append(f); changed = True
            word = "買い" if side == 1 else "売り"
            px = C[j]
            what = {"4h": "含み益が1R以上ある所から、4時間足の向きが逆に変わりました",
                    "shock_trend": "持っている向きと逆に急変 (普段の3.5倍以上) し、大きな流れも逆になりました",
                    "shock": f"持っている向きと逆に急変しました (普段の{z:.1f}倍)"}[f]
            urgent = act.startswith("すぐ")
            out.append((f"{'🚨' if urgent else '⚠️'} {act} USD/JPY ({word} {e:.3f})",
                        f"**{act}**\n{what}\n今 {px:.3f} ({side * (px - e) * 100:+.0f}pips) / 損切り {p['stop']:.3f}\n━━━━━━━━━━\n根拠: {why}"
                        + ("\n※注文 (IFD-OCO) の利確・損切りが入っていれば、成行で決済して注文を取り消してください" if urgent else ""),
                        15548997 if urgent else 16776960))
    if changed:
        positions.save(ps)
    return out
