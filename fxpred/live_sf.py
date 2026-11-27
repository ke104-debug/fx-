"""通知する作戦「急変に付いていく (流れの方向)」(利用者の承認 2026-11-27)。
23年の研究で一番根拠のあった作戦 (reports/overnight.md): 感度243通り中240プラス、作る期間で選んだ設定が確かめる期間で141/141プラス。
中身は lab.shock_follow (影の運用で記録していたもの) と同じ:
  1時間・3時間・24時間の動きが普段の3.5倍以上 (または24時間で2円以上) → その向きに次の足で入る。
  大きな流れ (20日平均の向き) と同じ向きだけ。月曜・金曜 (日本時間) と 10時〜翌1時以外は入らない。1日1回。
  損切り 1.5ATR・利確 3ATR・最長24時間
遅れると優位が消える (15分遅れで期待値が半分〜0) ので、5分監視が毎時の足の確定直後 (0〜12分) に確かめて送る。"""
import json
import os

import numpy as np
import pandas as pd

Z_TH, YEN, STOP_ATR, TGT_ATR, HOLD = 3.5, 2.0, 1.5, 3.0, 24


def signal(h: pd.DataFrame) -> dict | None:
    """最新の確定した1時間足で合図が出ていれば返す (lab.shock_follow の最後の足だけ版)"""
    from . import lab
    d = lab.prep(h.iloc[-900:])
    if "trend_up" not in d:
        d["trend_up"] = d.Close.ewm(span=480).mean().diff(24) > 0
    i = len(d) - 1
    r = d.Close.diff().values
    sd = pd.Series(r).rolling(480).std().values[i]
    c, A = d.Close.values, d.atr.values
    if np.isnan(sd) or sd <= 0 or i < 30:
        return None
    m1, m3, m24 = r[i], c[i] - c[i - 3], c[i] - c[i - 24]
    z = max(abs(m1) / sd, abs(m3) / (sd * np.sqrt(3)), abs(m24) / (sd * np.sqrt(24)))
    if not (z >= Z_TH or abs(m24) >= YEN):
        return None
    jst_end = d.index[i] + pd.Timedelta(hours=10)          # 足の終わりの日本時間 (lab と同じ)
    if not (jst_end.hour >= 10 or jst_end.hour < 1) or jst_end.weekday() in (0, 4):
        return None
    side = 1 if (m3 if m3 != 0 else m24) > 0 else -1
    if (side == 1) != bool(d.trend_up.iat[i]):
        return None
    return dict(side=side, entry=float(c[i]), stop=float(c[i] - side * STOP_ATR * A[i]), target=float(c[i] + side * TGT_ATR * A[i]),
                atr=float(A[i]), z=float(z), bar=f"{d.index[i]:%Y-%m-%d %H:%M}",
                note=f"1〜24時間の動きが普段の{z:.1f}倍 ({'上げ' if side == 1 else '下げ'})、大きな流れと同じ向き")


def check(st: dict, now_epoch: int, price: float):
    """5分監視から: 毎時0〜12分に1回だけ確かめる。合図があれば (title, body, color, pos) を返す"""
    now = pd.Timestamp(now_epoch, unit="s", tz="UTC")
    if now.minute > 12:
        return None
    slot = f"{now:%Y-%m-%d %H}"
    if st.get("sf_slot") == slot:
        return None
    st["sf_slot"] = slot
    from . import scan, regime as rg, fmt, positions, signal_audit
    h = rg.prepare(scan.load_recent())
    s = signal(h)
    if not s:
        print("(急変に付いていく) 合図なし")
        return None
    day = s["bar"][:10]
    if st.get("sf_day") == day:
        return None
    ok, why = signal_audit.rule_ok("SF", s["side"], h)
    if not ok:
        print("(合図の点検で止めた)", why)
        return None
    st["sf_day"] = day
    if abs(price - s["entry"]) > 0.6 * s["atr"]:      # 足の確定から大きく動いた後なら入らない (遅れの検証)
        scan.log_signal(now.floor("min"), "SF", s, 0, "見送り(動いた後)", sent=False, conditions="遅れ")
        print(f"(急変に付いていく) 確定から {abs(price - s['entry']) * 100:.0f}pips 動いたので見送り")
        return None
    rel = positions.conflict(s["side"])
    if rel == "same":
        return None
    try:
        ss = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config", "strategy_stats.json")))["stats"].get("SF", {})
    except Exception:
        ss = {}
    reasons = [s["note"], "23年の検証で一番根拠のある作戦 (設定を広く変えても崩れない)"]
    cautions = ["遅れるほど不利: 通知から15分を過ぎたら見送り"] + (["保有中の取引と逆向き"] if rel == "opposite" else [])
    title, body = fmt.signal(s["side"], price, s["stop"], s["target"], 60, reasons, cautions,
                             f"{(now + pd.Timedelta(hours=9)):%m/%d %H:%M} から24時間", None)
    body = body.replace("通知から60分を過ぎたら見送り", "通知から15分を過ぎたら見送り")
    if ss:
        body = body.replace("━━━━━━━━━━", f"過去の実績: この作戦は23年で{ss['n']}回・勝率{ss['win']:.0%}・1回平均{ss['R']:+.2f}R"
                            f" (2016年以降 {ss['R16']:+.2f}R)\n━━━━━━━━━━", 1)
    scan.log_signal(now.floor("min"), "SF", dict(s, entry=price), 60, "買い" if s["side"] == 1 else "売り", sent=True, conditions="急変|流れの方向")
    return title, body, 5763719, (s["side"], price, s["stop"], s["target"], "SF")
