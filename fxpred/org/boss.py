"""ボスAI (組織の最上位・利用者の指示 2026-11-27: 全統括AIが暴走しているように見える。監査と点検部隊が弱い)。
全統括AIの結論を最後に確かめ、決まりに合わなければ覆す。監査AI・点検部隊はボスAIに直接報告する。
判断は決まり (下の B1〜B6) で行う。AI (Groq/Gemini の無料枠) はボスの説明の1行と、見落としの指摘だけ (判断は変えない)。

B1 点検部隊が売買の判断に関わる問題を見つけている → 停止 (新しく入らない・合図を送らない)
B2 リスク統括が「警戒」 → 停止
B3 方向の根拠が1人の意見だけ (監査AI: 6割以上を1人が占める) → 向きを取り消す (様子見)
B4 方向が大きな流れと逆で、1時間・4時間・日足の向きがそろって流れの側にある → 向きを取り消す (11/27 の売り寄りの形)
B5 全統括AIの直近4週間の成績 (向きを出した時の24時間後の当たり) が、20回以上で5割未満 → 向きを取り消す (成績が戻るまで)
B6 合図は「検証済みの作戦 (config/live.json)」かつ「最終関門 (signal_audit.gate) を通った」ものだけ承認。
   全統括AIの向きと合図が逆でも、検証済みの作戦を優先する (ただし注意を書く)"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
LOG = os.path.join(ROOT, "journal", "boss.csv")


def supervisor_record(days=28) -> dict:
    """全統括AI (方向の会議) の成績: 向きを出した時 (|強さ|≥0.25) の24時間後の当たり"""
    try:
        o = pd.read_csv(os.path.join(ROOT, "journal", "org.csv"), usecols=["time_utc", "price", "overall"])
    except Exception:
        return {"n": 0, "hit": None}
    o["t"] = pd.to_datetime(o.time_utc, utc=True)
    o = o.sort_values("t").drop_duplicates("t")
    s = o.set_index("t").price
    o = o[o.t >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)]
    o = o[o.t + pd.Timedelta(hours=24) <= s.index[-1]]
    o = o[o.overall.abs() >= 0.25]
    if o.empty:
        return {"n": 0, "hit": None}
    f = np.array([s.asof(t + pd.Timedelta(hours=24)) for t in o.t]) - o.price.values
    hit = float((np.sign(o.overall.values) == np.sign(f)).mean())
    return {"n": int(len(o)), "hit": hit}


def _trend_view(h: pd.DataFrame) -> tuple[int, list[str]]:
    """大きな流れ (20日平均の向き) と、1時間・4時間・日足の平均線の向き。そろっていれば その向き (+1/-1)、でなければ 0"""
    up = bool(h.trend_up.iat[-1]) if "trend_up" in h else bool(h.Close.ewm(span=480).mean().diff(24).iat[-1] > 0)
    c = h.Close
    tf = {"1時間": c, "4時間": c.resample("4h").last().dropna(), "日足": c.resample("1D").last().dropna()}
    sides = {k: (1 if x.ewm(span=20).mean().iat[-1] > x.ewm(span=50).mean().iat[-1] else -1) for k, x in tf.items()}
    big = 1 if up else -1
    agree = all(v == big for v in sides.values())
    return (big if agree else 0), [f"大きな流れ {'上' if up else '下'}"] + [f"{k} {'上' if v == 1 else '下'}" for k, v in sides.items()]


def review(res: dict, h: pd.DataFrame) -> dict:
    """全統括AIの結論 (meeting.final の res) を確かめる。res に boss を足し、必要なら結論を覆す"""
    reasons, overrides = [], []
    verdict = "承認"
    direction = int(np.sign(res.get("overall", 0))) if abs(res.get("overall", 0)) >= 0.25 else 0
    # B1・B2: 止める
    blk = res.get("inspector_block") or []
    if blk:
        verdict = "停止"; overrides.append("点検部隊の報告: " + blk[0][:60])
    if res.get("risk", {}).get("level") == "警戒":
        verdict = "停止"; overrides.append("リスク統括が警戒")
    # B3: 1人の意見だけ
    if direction and any("1人の意見でほぼ決まっている" in x for x in res.get("audit", [])):
        direction = 0; overrides.append("方向の根拠が1人の意見だけ (監査AIの報告)")
    # B4: 流れと逆
    tv, tv_txt = _trend_view(h)
    if direction and tv and direction != tv:
        direction = 0; overrides.append("方向が大きな流れと逆で、1時間・4時間・日足もそろって流れの側 (" + " / ".join(tv_txt) + ")")
    # B5: 全統括AIの成績
    rec = supervisor_record()
    from . import mode
    if mode.simple() and direction:
        direction = 0; overrides.append("方向の会議は休止中 (統廃合後は採点だけ。向きは出さない)")
    if direction and rec["n"] >= 20 and rec["hit"] is not None and rec["hit"] < 0.5:
        direction = 0; overrides.append(f"全統括AIの直近4週間の向きの当たりが {rec['hit']:.0%} ({rec['n']}回) で5割未満")
    # 合図
    sig = res.get("market", {}).get("signal")
    if verdict != "停止" and sig is not None:
        reasons.append("合図は検証済みの作戦だけ承認 (最終関門で確認)")
    if not overrides:
        reasons.append("決まりに反する点なし")
    word = {1: "買い寄り", -1: "売り寄り", 0: "向きなし (様子見)"}[direction]
    if verdict != "停止" and overrides:
        verdict = "修正"        # 全統括AIの結論を覆した
    if verdict == "停止":
        concl = "新しく入らない (" + " / ".join(overrides)[:80] + ")"
    elif mode.simple():
        concl = ("作戦の合図あり (" + str(res.get("market", {}).get("text", ""))[:50] + ")") if sig is not None else "様子見 (検証済みの作戦の合図なし)"
    elif overrides and res.get("verdict") in ("買い寄り", "売り寄り") and direction == 0:
        concl = f"様子見 (全統括AIの『{res.get('verdict')}』を取り消し: {overrides[-1][:60]})"
    else:
        concl = res.get("conclusion", "")
    risk_names = sorted({who for who, _ in res.get("risk", {}).get("items", [])})
    out = {"risk_names": risk_names, "verdict": verdict, "direction": direction, "word": word, "overrides": overrides, "reasons": reasons,
           "record": rec, "trend": tv_txt, "conclusion": concl}
    return out


def line(b: dict) -> str:
    rec = b.get("record", {})
    r = f"全統括AIの成績 {rec['hit']:.0%} ({rec['n']}回)" if rec.get("hit") is not None else "全統括AIの成績はまだ集計中"
    return (f"👑 ボスAI: {b['verdict']}・{b['word']}"
            + ("".join(f"\n　・覆した理由: {x}" for x in b["overrides"]) if b["overrides"] else "")
            + f"\n　・{r}")


def log(b: dict, res: dict):
    row = {"time_utc": f"{pd.Timestamp.now(tz='UTC').floor('30min'):%Y-%m-%d %H:%M}", "supervisor": res.get("verdict", ""),
           "supervisor_overall": round(float(res.get("overall", 0)), 3), "boss": b["verdict"], "direction": b["direction"],
           "overrides": " / ".join(b["overrides"])[:300], "risk_names": "|".join(b.get("risk_names", [])), "record_hit": b["record"].get("hit"), "record_n": b["record"].get("n")}
    old = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame()
    if len(old) and (old.time_utc == row["time_utc"]).any():
        return
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    pd.concat([old, pd.DataFrame([row])]).to_csv(LOG, index=False)
