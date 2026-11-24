"""監査AI: 組織の外から、遠慮なく問題点を挙げる担当。良い話はしない。
- 自動実行が止まっていないか / データが古くないか
- 送った通知が実際に勝っているか (負けていれば正直に言う)
- 組織の判断が実際に当たっているか、判断に使える担当がいない部隊はないか
- Gemini (無料枠) が失敗していないか
- 調整のしすぎ (過去に合わせ込み) の危険がないか"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def audit(h: pd.DataFrame, now=None) -> list[str]:
    now = now or pd.Timestamp.now(tz="UTC")
    out = []
    # データの鮮度 (平日)
    age = (now - h.index[-1]).total_seconds() / 3600
    if now.weekday() < 5 and age > 3:
        out.append(f"データが{age:.0f}時間更新されていない。今の判断は古い値にもとづいている")
    # 自動実行
    try:
        from ..dailyplan import run_health
        hl = run_health()
        if hl and ("❌" in hl or "⚠️" in hl):
            out.append("自動実行が足りていない: " + hl.replace("稼働 (直近24時間): ", ""))
    except Exception:
        pass
    # 送った通知の実績 (直近4週間)
    try:
        from ..periodic import evaluate_signals
        ev = evaluate_signals(h)
        if len(ev):
            ev["t"] = pd.to_datetime(ev.time_utc, utc=True)
            s = ev[(ev.sent.astype(str) == "True") & ev.pips.notna() & (ev.t >= now - pd.Timedelta(days=28))]
            k = ev[(ev.sent.astype(str) == "False") & ev.pips.notna() & (ev.t >= now - pd.Timedelta(days=28))]
            if len(s) >= 5 and s.pips.sum() < 0:
                out.append(f"送った通知は直近4週間で負けている ({len(s)}回・合計{s.pips.sum():+.0f}pips)")
            if len(s) >= 3 and len(k) >= 5 and k.pips.mean() > s.pips.mean() + 3:
                out.append(f"見送った合図の方が成績が良い (見送り 平均{k.pips.mean():+.1f}pips / 送った {s.pips.mean():+.1f}pips)。通知の基準がずれている")
            if len(s) < 2:
                out.append("直近4週間に送った通知がほとんどない。チャンスを逃していないか確認が必要")
    except Exception:
        pass
    # 組織
    try:
        W = json.load(open(os.path.join(ROOT, "config", "org_weights.json"))).get("weights", {})
        from .meeting import DEPTS
        units = {}
        for k, v in W.items():
            units.setdefault(DEPTS.get(k.split("|")[0], "?"), []).append(v)
        for u, ws in units.items():
            n = sum(w > 0 for w in ws)
            if n <= 1:
                out.append(f"{u}は判断に使える担当が{n}人だけ ({len(ws)}人中)。部隊の判断はほぼ1つの指標に頼っている")
        from .advisor import live_scores
        ls = live_scores()
        top = ls[ls.name == "統括AI (強さ0.25以上)"] if len(ls) else ls
        if len(top) and top.iloc[0].n >= 30 and top.iloc[0].pips < 0:
            out.append(f"統括AIの判断は実際の記録で外れている ({top.iloc[0].n}回・平均{top.iloc[0].pips:+.1f}pips)")
    except Exception:
        pass
    # Gemini
    try:
        g = json.load(open(os.getenv("FX_GEMINI_STATE", "state/gemini.json")))
        if g.get("last_ts") and pd.Timestamp.now().timestamp() - g["last_ts"] > 6 * 3600:
            out.append("Gemini の情報AIが6時間以上答えていない (無料枠の上限か、失敗が続いている)")
    except Exception:
        if os.getenv("GEMINI_API_KEY"):
            out.append("Gemini の情報AIの記録がない")
    # 合わせ込みの危険
    try:
        dec = pd.read_csv(os.path.join(ROOT, "journal", "org_decisions.csv"))
        dec["date"] = pd.to_datetime(dec.date)
        n = (dec.date >= (now.tz_localize(None) - pd.Timedelta(days=28))).sum()
        if n >= 6:
            out.append(f"直近4週間で担当の重みを{n}回変えている。変えすぎると過去に合わせ込むだけになる")
    except Exception:
        pass
    return out or ["大きな問題は見つからない (ただし記録が少ないうちは判断できないことも多い)"]


if __name__ == "__main__":
    from .. import scan
    print("\n".join(audit(scan.load_recent())))
