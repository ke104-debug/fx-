"""夜の復習 (運用時間外: 日本時間 2時台に1回)。その日の取引時間 (10時〜翌1時) を振り返り、細かく再現して、
「どんな場面で、どうすれば成果を上げられたか」を記録する。ルールは変えない (凍結中)。

1. 事実を集める: 値動き (大きく動いた時間)、合図 (送った・見送った・その後)、影の運用の作戦、候補の作戦 (指標の時間を外した急変・東京朝のブレイク)、
   指標の実況シミュレーション、AI の見立ての当たり (会議の向き・5分監視の Groq・15分の Gemini)
2. Gemini と Groq にそれぞれ反省させる: 良かった点・悪かった点・成果を上げられた場面と取り方・仮説 (条件 → 結果、確かめ方)
3. 記録: journal/nightly_review.csv (1日1行) と journal/hypotheses.csv (仮説を1つ1行)。
   週次の見直しで仮説をまとめ、何度も出てくる仮説は「提案」として改善研究部隊へ (23年のデータで確かめてから判断)
朝8時の報告には2行だけ載せる。"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
J = os.path.join(ROOT, "journal")
REVIEW = os.path.join(J, "nightly_review.csv")
HYPO = os.path.join(J, "hypotheses.csv")


def _read(name):
    p = os.path.join(J, name)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def window(now=None):
    """直前の取引時間 (日本時間 前日10時〜当日1時) を UTC で"""
    now = now or pd.Timestamp.now(tz="UTC")
    jst = now + pd.Timedelta(hours=9)
    day = jst.normalize() - (pd.Timedelta(days=1) if jst.hour < 10 else pd.Timedelta(0))
    t0 = day + pd.Timedelta(hours=10) - pd.Timedelta(hours=9)
    return t0, t0 + pd.Timedelta(hours=15), f"{day:%Y-%m-%d}"


def facts(h: pd.DataFrame, t0, t1) -> list[str]:
    F = []
    w = h[(h.index >= t0) & (h.index < t1)]
    if not len(w):
        return ["この時間の値動きの記録なし"]
    F.append(f"値動き: {w.Open.iat[0]:.2f} → {w.Close.iat[-1]:.2f} ({(w.Close.iat[-1] - w.Open.iat[0]) * 100:+.0f}pips)、高値 {w.High.max():.2f} 安値 {w.Low.min():.2f} (値幅 {(w.High.max() - w.Low.min()) * 100:.0f}pips)")
    mv = (w.Close - w.Open) * 100
    for t, v in mv.abs().sort_values(ascending=False).head(3).items():
        F.append(f"大きく動いた1時間: {(t + pd.Timedelta(hours=9)):%H時} {mv[t]:+.0f}pips")
    # 合図
    try:
        from .periodic import evaluate_signals
        ev = evaluate_signals(h)
        ev["t"] = pd.to_datetime(ev.time_utc, utc=True)
        d = ev[(ev.t >= t0) & (ev.t < t1)]
        for r in d.itertuples():
            F.append(f"合図: {(r.t + pd.Timedelta(hours=9)):%H:%M} {r.strategy} {'買い' if r.side == 1 else '売り'} "
                     f"{'送った' if str(r.sent) == 'True' else '見送り(' + str(r.verdict) + ')'} → {r.result} {r.pips if r.pips == r.pips else ''}pips")
        if not len(d):
            F.append("合図: なし")
    except Exception as e:
        F.append(f"合図の答え合わせができず ({e})")
    # 影の運用と候補の作戦
    try:
        from . import lab, shadow
        dd = lab.prep(h.iloc[-1500:])
        for name, c in shadow.CANDIDATES.items():
            t = c["fn"](dd, **c["params"])
            t = t[(t.time >= t0) & (t.time < t1)] if len(t) else t
            if len(t):
                pp = t.pips if "pips" in t else t.R * t.risk
                F.append(f"影の作戦「{name[:18]}」: {len(t)}回 {pp.sum():+.0f}pips")
        sf = lab.shock_follow(dd, z_th=3.5)
        sf = sf[(sf.time >= t0) & (sf.time < t1)] if len(sf) else sf
        if len(sf):
            ny = sf.time.dt.tz_convert("America/New_York").dt.hour
            F.append(f"候補「急変に付いていく (米国の指標の時間を外す)」: {(~ny.isin([7, 8, 9, 10])).sum()}回 / 外さない場合 {len(sf)}回")
    except Exception as e:
        F.append(f"影の作戦の再現ができず ({str(e)[:40]})")
    # 指標の実況シミュレーション
    es = _read("eventsim.csv")
    if len(es):
        es["t"] = pd.to_datetime(es.release_utc, utc=True)
        for r in es[(es.t >= t0) & (es.t < t1 + pd.Timedelta(hours=1))].itertuples():
            F.append(f"指標 {str(r.events)[:24]}: 発表後60分 {r.move_60:+.0f}pips / 順張り {r.A} / 逆張り {r.B} / 組織 {r.C}")
    # AI の見立ての当たり
    c = h.Close
    og = _read("org.csv")
    if len(og):
        og["t"] = pd.to_datetime(og.time_utc, utc=True)
        o = og[(og.t >= t0) & (og.t < t1)]
        fut = c.reindex(o.t + pd.Timedelta(hours=4), method="nearest", tolerance=pd.Timedelta(minutes=45)).values
        k = (o.overall.abs() >= 0.25) & pd.notna(fut)
        if k.sum():
            F.append(f"会議の向き (4時間後): {int(k.sum())}回中 {((o.overall[k] > 0) == (fut[k] > o.price[k])).mean():.0%} 当たり")
        else:
            F.append("会議の向き: ほぼ中立 (向きを出さなかった)")
    qa = _read("quick_ai.csv")
    if len(qa):
        qa["t"] = pd.to_datetime(qa.time_utc, utc=True)
        q = qa[(qa.t >= t0) & (qa.t < t1)]
        fut = c.reindex(q.t + pd.Timedelta(minutes=30), method="nearest", tolerance=pd.Timedelta(minutes=35)).values
        for col, who in (("view30", "Groq (5分ごと)"), ("g_view30", "Gemini (15分ごと)")):
            if col in q:
                k = q[col].notna() & (q[col].abs() >= 0.3) & pd.notna(fut)
                if k.sum() >= 3:
                    F.append(f"{who} の30分後の見立て: {int(k.sum())}回中 {((q[col][k] > 0) == (fut[k] > q.price[k])).mean():.0%} 当たり")
    return F


PROMPT = ("あなたは FX (ドル円) の売買判断チームの反省会の担当。以下は今日の取引時間 (日本時間10時〜翌1時) の事実。"
          "事実だけを根拠に、しっかり反省する。良かった点・悪かった点を具体的に。成果を上げられた場面があれば、"
          "どの時間に、どんな条件がそろっていて、どうすれば取れたかを書く。仮説は「条件 → 結果」の形で、23年の過去データで確かめられる形にする。"
          "ルールの変更は提案しない (記録をためてから判断する)。JSON だけ: "
          '{"良かった": ["", ""], "悪かった": ["", ""], "成果を上げられた場面": [{"時間": "", "内容": "", "取り方": ""}], '
          '"仮説": ["条件 → 結果 (確かめ方)"], "一言": "40字以内"}\n\n')


def run(h: pd.DataFrame, now=None) -> dict:
    t0, t1, day = window(now)
    F = facts(h, t0, t1)
    out = {"date": day, "facts": F}
    try:
        from .org import gem, groq
        g = gem.call("夜の復習", PROMPT + "\n".join(F))
        q = groq.call("夜の復習", PROMPT + "\n".join(F))
        out["gemini"], out["groq"] = (g if isinstance(g, dict) else {}), (q if isinstance(q, dict) else {})
    except Exception as e:
        print("夜の復習の AI のエラー:", e)
        out["gemini"], out["groq"] = {}, {}
    row = {"date": day, "facts": " | ".join(F), "gemini": json.dumps(out["gemini"], ensure_ascii=False)[:3000],
           "groq": json.dumps(out["groq"], ensure_ascii=False)[:3000]}
    old = _read("nightly_review.csv")
    if not len(old) or day not in set(old.date.astype(str)):
        os.makedirs(J, exist_ok=True)
        pd.concat([old, pd.DataFrame([row])]).to_csv(REVIEW, index=False)
        hy = [{"date": day, "by": who, "hypothesis": str(x)[:150]} for who, r in (("Gemini", out["gemini"]), ("Groq", out["groq"]))
              for x in (r.get("仮説") or []) if str(x).strip()]
        if hy:
            pd.DataFrame(hy).to_csv(HYPO, mode="a", header=not os.path.exists(HYPO), index=False)
    print(f"(夜の復習) {day}: 事実 {len(F)}件 / 仮説 Gemini {len(out['gemini'].get('仮説', []) or [])} Groq {len(out['groq'].get('仮説', []) or [])}")
    return out


def morning_lines() -> list[str]:
    """朝8時の報告用に2行"""
    d = _read("nightly_review.csv")
    if not len(d):
        return []
    r = d.iloc[-1]
    try:
        g = json.loads(r.gemini) if isinstance(r.gemini, str) else {}
        q = json.loads(r.groq) if isinstance(r.groq, str) else {}
    except Exception:
        g, q = {}, {}
    one = g.get("一言") or q.get("一言") or ""
    sc = (g.get("成果を上げられた場面") or q.get("成果を上げられた場面") or [{}])[0]
    L = [f"昨夜の復習 ({r.date}): {str(one)[:50]}"]
    if isinstance(sc, dict) and sc.get("内容"):
        L.append(f"取れた場面: {str(sc.get('時間', ''))[:8]} {str(sc.get('内容', ''))[:40]} → {str(sc.get('取り方', ''))[:40]}")
    return L


def weekly_hypotheses() -> list[str]:
    """週次: 直近4週間の仮説を AI にまとめさせ、何度も出てくるものを挙げる (改善研究部隊の提案の候補)"""
    d = _read("hypotheses.csv")
    if not len(d):
        return ["仮説の記録なし"]
    d = d[pd.to_datetime(d.date) >= pd.Timestamp.now() - pd.Timedelta(days=28)]
    try:
        from .org import groq
        r, who = groq.ask("仮説のまとめ", "FX の夜の復習で出た仮説の一覧。似たものをまとめ、出てきた日数の多い順に最大5つ、"
                          "それぞれ「仮説 (日数)」で。JSON だけ: {\"top\": [\"\"]}\n\n" + "\n".join(f"{x.date} {x.hypothesis}" for x in d.itertuples()))
        if isinstance(r, dict) and r.get("top"):
            return [f"{str(x)[:90]}" for x in r["top"][:5]] + [f"(まとめ: {who}。3日以上出たものは、Claude が23年のデータで確かめて提案にする)"]
    except Exception as e:
        print("仮説のまとめのエラー:", e)
    return [f"仮説 {len(d)}件 (まとめられず)"]
