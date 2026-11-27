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
    oi = open_issues()
    r = f"全統括AIの成績 {rec['hit']:.0%} ({rec['n']}回)" if rec.get("hit") is not None else "全統括AIの成績はまだ集計中"
    return (f"👑 ボスAI: {b['verdict']}・{b['word']}"
            + ("".join(f"\n　・覆した理由: {x}" for x in b["overrides"]) if b["overrides"] else "")
            + f"\n　・{r}"
            + (("\n　・未解決の問題: " + " / ".join(oi[:4])) if oi else ""))


def log(b: dict, res: dict):
    row = {"time_utc": f"{pd.Timestamp.now(tz='UTC').floor('30min'):%Y-%m-%d %H:%M}", "supervisor": res.get("verdict", ""),
           "supervisor_overall": round(float(res.get("overall", 0)), 3), "boss": b["verdict"], "direction": b["direction"],
           "overrides": " / ".join(b["overrides"])[:300], "risk_names": "|".join(b.get("risk_names", [])), "record_hit": b["record"].get("hit"), "record_n": b["record"].get("n")}
    old = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame()
    if len(old) and (old.time_utc == row["time_utc"]).any():
        return
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    pd.concat([old, pd.DataFrame([row])]).to_csv(LOG, index=False)


# ---------------------------------------------------------------- 未解決の問題の管理 (2026-11-27 利用者の指摘 K006)
# 点検部隊の NG を台帳 (journal/issues.csv) に載せ、直るまで追いかける。放置させないのがボスAIの仕事。
#  初めての NG → すぐ知らせる / 6時間たっても直らない → 6時間ごとに督促 / 24時間以上 → 『🛑 未解決』として毎回の報告の先頭に出し続ける
#  直ったら (OK に戻ったら) 解決として記録し、知らせる。Claude は作業の最初にこの台帳の未解決を確かめる (CLAUDE.md)
ISSUES = os.path.join(ROOT, "journal", "issues.csv")
COLS = ["check", "first_utc", "last_utc", "count", "detail", "status", "resolved_utc", "last_notice_utc"]


def _load_issues() -> pd.DataFrame:
    try:
        d = pd.read_csv(ISSUES, dtype=str).fillna("")
        for c in COLS:
            if c not in d:
                d[c] = ""
        return d[COLS].astype(str)
    except Exception:
        return pd.DataFrame(columns=COLS, dtype=str)


def track_issues(rows: list[dict], deep=False, dry=False) -> list[tuple[str, str, str]]:
    now = pd.Timestamp.now(tz="UTC")
    d = _load_issues()
    msgs, new, nag, fixed = [], [], [], []
    for r in rows:
        openm = (d.check == r["check"]) & (d.status == "未解決")
        if not r["ok"]:
            if openm.any():
                i = d.index[openm][-1]
                d.loc[i, ["last_utc", "detail"]] = [f"{now:%Y-%m-%d %H:%M}", str(r["detail"])[:300]]
                d.loc[i, "count"] = str(int(float(d.loc[i, "count"] or 0)) + 1)
                age = (now - pd.Timestamp(d.loc[i, "first_utc"], tz="UTC")).total_seconds() / 3600
                last = pd.Timestamp(d.loc[i, "last_notice_utc"], tz="UTC") if str(d.loc[i, "last_notice_utc"]) not in ("", "nan") else now - pd.Timedelta(days=1)
                if age >= 6 and (now - last) >= pd.Timedelta(hours=6):
                    nag.append((r["check"], age, r["detail"]))
                    d.loc[i, "last_notice_utc"] = f"{now:%Y-%m-%d %H:%M}"
            else:
                d.loc[len(d)] = [r["check"], f"{now:%Y-%m-%d %H:%M}", f"{now:%Y-%m-%d %H:%M}", "1", str(r["detail"])[:300], "未解決", "", f"{now:%Y-%m-%d %H:%M}"]
                new.append((r["check"], r["detail"]))
        elif openm.any():
            # 毎晩だけの点検 (deep) は毎時の回では見ないので、OK になっても解決にしない
            i = d.index[openm][-1]
            d.loc[i, ["status", "resolved_utc"]] = ["解決", f"{now:%Y-%m-%d %H:%M}"]
            fixed.append(r["check"])
    if new:
        msgs.append(("wait", "🔧 点検部隊: 新しい問題 (ボスAIが追跡を開始)",
                     "\n".join(f"・**{c}**: {str(t)[:250]}" for c, t in new)
                     + "\n━━━━━━━━━━\n直るまでボスAIが追いかけます (6時間ごとに督促、24時間を超えたら報告の先頭に出し続けます)"))
    if nag:
        msgs.append(("wait", "🛑 ボスAI: 未解決の問題の督促",
                     "\n".join(f"・**{c}** — {age:.0f}時間 未解決: {str(t)[:200]}" for c, age, t in nag)
                     + "\n━━━━━━━━━━\n修正を Claude に依頼してください (Claude は作業の最初にこの一覧を確認する決まり)"))
    if fixed:
        msgs.append(("info", "✅ ボスAI: 問題が解決しました", "\n".join(f"・{c}" for c in fixed)))
    if not dry:
        os.makedirs(os.path.dirname(ISSUES), exist_ok=True)
        d.to_csv(ISSUES, index=False)
    return msgs


def open_issues() -> list[str]:
    d = _load_issues()
    d = d[d.status == "未解決"]
    now = pd.Timestamp.now(tz="UTC")
    out = []
    for r in d.itertuples():
        age = (now - pd.Timestamp(r.first_utc, tz="UTC")).total_seconds() / 3600
        out.append(f"{'🛑' if age >= 24 else '⚠️'} {r.check} ({age:.0f}時間 未解決)")
    return out
