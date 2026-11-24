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


def meeting_check(res: dict) -> list[str]:
    """会議の中身の監査: 結論が根拠と食い違っていないか、古い情報・対立・偏りを見落としていないか"""
    out = []
    if not res:
        return out
    ops = res.get("ops", [])
    if res.get("market", {}).get("conflict"):
        out.append("作戦の合図と方向の会議が対立したまま結論を出している。対立時の成績はまだ検証中")
    voters = [o for g in res.get("depts", {}).values() for o in g.get("voters", [])]
    if voters:
        top = max(voters, key=lambda o: abs(o["bias"]))
        share = abs(top["bias"]) / max(sum(abs(o["bias"]) for o in voters), 1e-9)
        if share > 0.6 and abs(res.get("overall", 0)) >= 0.25:
            out.append(f"結論が「{top['name'].split('|')[-1]}」1人の意見でほぼ決まっている ({share:.0%})")
    summ = [o for o in ops if o["name"].startswith("AI情報|") and o["name"] not in ("AI情報|まとめ", "AI情報|介入けん制の段階")]
    if summ and abs(res.get("overall", 0)) >= 0.25:
        ai = sum(o["bias"] for o in summ)
        if ai * res["overall"] < 0 and abs(ai) >= 1:
            out.append("ニュースを読んだAIの見立てと、会議の結論が逆向き (どちらかが古いか、材料を読み違えている可能性)")
    if res.get("stale"):
        out.append("古い情報を外して判断した: " + " / ".join(res["stale"][:3]))
    if res.get("secs", 0) > 120:
        out.append(f"会議に{res['secs']:.0f}秒かかっている (情報処理の遅れ)")
    return out


def audit(h: pd.DataFrame, now=None, res: dict | None = None) -> list[str]:
    """res に会議の結果 (meeting.final) を渡すと、会議の中身も監査する"""
    now = now or pd.Timestamp.now(tz="UTC")
    out = meeting_check(res)
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
    # 自己正当化の防止: 採用した新ルールに「使っていない期間での検証」の記録があるか
    try:
        pr = pd.read_csv(os.path.join(ROOT, "journal", "proposals.csv"))
        bad = pr[(pr.status == "採用") & (pr.holdout.isna() | (pr.holdout.astype(str).str.strip() == ""))]
        if len(bad):
            out.append(f"使っていない期間で確かめずに採用したルールがある: {', '.join(bad.title.astype(str).head(3))}")
        slow = pr[pr.status.isin(["提案", "検証中"])]
        if len(slow) >= 5:
            out.append(f"検証待ちの提案が{len(slow)}件たまっている")
    except Exception:
        pass
    # 通知の遅れ
    try:
        lt = pd.read_csv(os.path.join(ROOT, "journal", "latency.csv"))
        lt = lt.tail(50)
        if len(lt) >= 5 and lt.delay_min.median() > 15:
            out.append(f"合図が出てから通知までの遅れが中央値{lt.delay_min.median():.0f}分 (足が確定してから)")
    except Exception:
        pass
    return out or ["大きな問題は見つからない (ただし記録が少ないうちは判断できないことも多い)"]


if __name__ == "__main__":
    from .. import scan
    print("\n".join(audit(scan.load_recent())))
