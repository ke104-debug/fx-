"""週次・月次の見直し (毎週土曜10時 / 毎月最初の土曜10時、日本時間)

  集計: 送ったシグナルの勝率・損益 (pips)、信頼度の帯ごとの成績、送らなかったシグナルの「もし入っていたら」の成績、
        毎朝の予想の当たり (方向・値幅・大きなずれ)
  調整: 件数が十分ある時だけ、小さく動かす。調整は journal/learning.json に保存し、監視がそのまま使う
    週次 … 毎時の分析の通知の基準点 (send_at) を ±5 点だけ動かせる (範囲 55〜85)
    月次 … 条件ごとの配点 (cond_weights) を、成績の良い条件は +5・悪い条件は -5 (各 0〜45、合計100に正規化)
"""
import json
import os
from datetime import timedelta

import numpy as np
import pandas as pd

from . import dailyplan as dp
from . import edge

SIGNALS = os.path.join(dp.J, "signals.csv")
HOLD_H = {"H3": 14, "H4": 48, "H5": 24, "S": 72}


def evaluate_signals(h: pd.DataFrame) -> pd.DataFrame:
    """記録したシグナルを、実際の値動きで判定 (利確 / 損切り / 期限切れ)"""
    if not os.path.exists(SIGNALS):
        return pd.DataFrame()
    s = pd.read_csv(SIGNALS)
    out = []
    for r in s.itertuples():
        t0 = pd.Timestamp(r.time_utc, tz="UTC") + timedelta(hours=1)
        side, e, stop = int(r.side), float(r.entry), float(r.stop)
        tgt = float(r.target) if str(r.target) not in ("", "nan") else e + side * 2 * abs(e - stop)
        w = h[h.index >= t0].iloc[:HOLD_H.get(getattr(r, "key", "H4"), 48)]
        if len(w) == 0:
            out.append({**r._asdict(), "result": "判定待ち", "pips": np.nan}); continue
        res, px = None, w.Close.iat[-1]
        for b in w.itertuples():
            if (side == 1 and b.Low <= stop) or (side == -1 and b.High >= stop): res, px = "損切り", stop; break
            if (side == 1 and b.High >= tgt) or (side == -1 and b.Low <= tgt): res, px = "利確", tgt; break
        if res is None:
            res = "期限切れ" if len(w) >= HOLD_H.get(getattr(r, "key", "H4"), 48) else "判定待ち"
        out.append({**r._asdict(), "result": res, "pips": round(side * (px - e) * 100, 1) if res != "判定待ち" else np.nan})
    return pd.DataFrame(out)


def _stats(x: pd.DataFrame) -> str:
    x = x[x.result != "判定待ち"]
    if x.empty:
        return "0回"
    return f"{len(x)}回 勝率{(x.pips > 0).mean():.0%} 合計{x.pips.sum():+.0f}pips 平均{x.pips.mean():+.1f}"


def review(h: pd.DataFrame, days: int, monthly: bool) -> tuple[str, str]:
    learn = dp.load_learning()
    since = h.index[-1] - timedelta(days=days)
    ev = evaluate_signals(h)
    lines, changes = [], []
    if not ev.empty:
        ev["t"] = pd.to_datetime(ev.time_utc, utc=True)
        ev = ev[ev.t >= since]
    sent = ev[ev.sent.astype(str) == "True"] if not ev.empty else ev
    skipped = ev[ev.sent.astype(str) == "False"] if not ev.empty else ev
    lines += ["**シグナルの成績**", f"送った: {_stats(sent) if len(sent) else '0回'}",
              f"見送った (もし入っていたら): {_stats(skipped) if len(skipped) else '0回'}"]
    if len(sent):
        b = pd.cut(sent.confidence, [0, 64, 74, 100], labels=["55〜64%", "65〜74%", "75%以上"])
        for lab, g in sent.groupby(b, observed=True):
            lines.append(f"・信頼度{lab}: {_stats(g)}")
    # 影の運用 (通知していない新しい作戦) の成績
    try:
        from . import lab, shadow
        d = lab.prep(h.iloc[-(days * 24 + 900):])
        for name, c in shadow.CANDIDATES.items():
            t = c["fn"](d, **c["params"])
            t = t[t.time >= since]
            if len(t) and "pips" not in t:          # 損益を R (損切り幅=1) で返す作戦は pips に直す
                t = t.assign(pips=t.R * t.risk)
            lines.append(f"影の運用「{name}」: " + (f"{len(t)}回 勝率{(t.pips > 0).mean():.0%} 合計{t.pips.sum():+.0f}pips 平均{t.pips.mean():+.1f}" if len(t) else "0回"))
    except Exception as e:
        lines.append(f"影の運用: 集計できず ({e})")
    # 毎朝の予想
    if os.path.exists(dp.REVIEW):
        rv = pd.read_csv(dp.REVIEW)
        rv = rv[pd.to_datetime(rv.time_utc, utc=True) >= since]
        if len(rv):
            lines += ["", "**毎朝の予想**",
                      f"方向 {rv.hit_dir.mean():.0%} / 値幅 {rv.in_range.mean():.0%} / 大きなずれ {(rv.big_miss.fillna('') != '').sum()}日 ({len(rv)}日中)"]
    # ---- 調整 (件数が足りない時は変えない)
    done = sent[sent.result != "判定待ち"] if len(sent) else sent
    sk = skipped[skipped.result != "判定待ち"] if len(skipped) else skipped
    send_at = learn.get("send_at", edge.SEND_AT)
    low = done[done.confidence < send_at + 10] if len(done) else done
    if len(low) >= 8 and low.pips.sum() < 0:
        learn["send_at"] = min(85, send_at + 5)
        changes.append(f"基準ぎりぎりの通知がマイナス ({_stats(low)}) → 通知の基準を {send_at}% → {learn['send_at']}% に上げる")
    near = sk[sk.confidence >= send_at - 10] if len(sk) else sk
    if len(near) >= 8 and near.pips.sum() > 0 and (near.pips > 0).mean() >= 0.5:
        learn["send_at"] = max(55, send_at - 5)
        changes.append(f"見送ったシグナルがプラス ({_stats(near)}) → 通知の基準を {send_at}% → {learn['send_at']}% に下げる")
    if monthly and len(done) >= 20:
        w = dict(learn.get("cond_weights") or edge.COND_WEIGHTS)
        for cond in w:
            has = done[done.conditions.fillna("").str.contains(cond)]
            not_ = done[~done.conditions.fillna("").str.contains(cond)]
            if len(has) >= 10 and len(not_) >= 10:
                diff = has.pips.mean() - not_.pips.mean()
                if diff > 3: w[cond] = min(60, w[cond] + 5); changes.append(f"「{cond}」がそろった時の成績が良い ({diff:+.1f}pips) → 配点+5")
                if diff < -3: w[cond] = max(-30, w[cond] - 5); changes.append(f"「{cond}」がそろっても成績が悪い ({diff:+.1f}pips) → 配点-5")
        learn["cond_weights"] = w
    if not changes:
        need = "20回以上 (月次)" if monthly else "8回以上"
        changes.append(f"変更なし (判断に必要な件数 {need} に届いていない、または成績に偏りがない)")
    lines += ["", "**調整**"] + [f"・{c}" for c in changes]
    lines += [f"現在の通知の基準: {learn.get('send_at', edge.SEND_AT)}% / 配点: " +
              " ".join(f"{k}{v}" for k, v in (learn.get("cond_weights") or edge.COND_WEIGHTS).items())]
    learn.setdefault("history", []).append({"date": f"{pd.Timestamp.now(tz=dp.JST):%Y-%m-%d}",
                                            "type": "月次" if monthly else "週次", "changes": changes})
    learn["history"] = learn["history"][-24:]
    os.makedirs(dp.J, exist_ok=True)
    json.dump(learn, open(dp.LEARN, "w"), ensure_ascii=False, indent=1)
    try:   # 進言AI (AIの組織の足りない部分・弱っている部分)
        from .org import advisor, auditor, review as org_review
        lines += ["", "**改善研究部隊・進言AI**"] + [f"・{x}" for x in advisor.advise()]
        lines += ["", "**シミュレーション・検証部隊 (進言の再検証)**"] + [f"・{x}" for x in org_review.run()]
        from .org import aide
        lines += ["", "**統括補佐AI の答え合わせ**"] + [f"・{x}" for x in aide.score(h)]
        from .org import quick_ai
        lines += [f"・{x}" for x in quick_ai.score(h)]
        from .org import postmortem
        lines += ["", "**原因究明部隊**"] + [f"・{x}" for x in postmortem.run(h)]
        try:
            from .org import gem
            ai = gem.call("週次の振り返り", "FX (ドル円) の自動売買チームの週次の振り返り。以下の記録だけを根拠に、来週試すべき改善案を3つ、"
                          "それぞれ40字以内で出す。過去に合わせ込むだけの案は避け、確かめ方も一言添える。JSON だけ: {\"ideas\": [\"\", \"\", \"\"]}\n\n"
                          + "\n".join(lines)[-3000:])
            if isinstance(ai, dict) and ai.get("ideas"):
                lines += ["", "**改善案 (Gemini。提案として検証部隊へ)**"] + [f"・{str(x)[:80]}" for x in ai["ideas"][:3]]
                from .org import proposals
                for x in ai["ideas"][:3]:
                    proposals.add("改善研究部隊 (Gemini)", str(x)[:80], built_on="週次の記録", status="提案")
        except Exception as e:
            print("週次の改善案のエラー:", e)
        _res = _meeting_safe(h)
        _found = auditor.audit(h, res=_res)
        try:   # 検証部隊の補助 (Groq): 提案の一覧を読み、過去への合わせ込みの危険が高いものを指摘
            from .org import groq, proposals
            pr = proposals.load()
            pr = pr[pr.status.isin(["提案", "候補"])].tail(15)
            if len(pr):
                r, who = groq.ask("検証の補助", "FX の売買ルールの提案の一覧。過去への合わせ込み (たまたまの結果を拾っている) の危険が高いものと、"
                                  "確かめ方が足りないものを最大3つ、各50字以内で指摘する。JSON だけ: {\"issues\": [\"\"]}\n\n"
                                  + "\n".join(f"{x.id} {x.title} / 結果: {x.holdout} / 状態: {x.status}" for x in pr.itertuples()))
                if isinstance(r, dict) and r.get("issues"):
                    lines += ["", f"**検証部隊の補助 ({who})**"] + [f"・{str(x)[:80]}" for x in r["issues"][:3]]
        except Exception as e:
            print("検証の補助のエラー:", e)
        try:
            from . import nightly
            lines += ["", "**夜の復習で何度も出た仮説 (直近4週間)**"] + [f"・{x}" for x in nightly.weekly_hypotheses()]
        except Exception as e:
            print("仮説のまとめのエラー:", e)
        lines += ["", "**監査AI (問題点)**"] + [f"・{x}" for x in _found + auditor.ai_audit(_found, _res)]
    except Exception as e:
        print("進言AIのエラー:", e)
    title = f"📊 {'月次' if monthly else '週次'}の見直し USD/JPY ({(since + timedelta(days=1)).tz_convert(dp.JST):%m/%d}〜{h.index[-1].tz_convert(dp.JST):%m/%d})"
    return title, "\n".join(lines)


def _meeting_safe(h):
    try:
        from .org import meeting
        return meeting.final(h)
    except Exception:
        return None


def run(monthly=False, dry=False):
    from . import notify, regime as rg, scan
    h = rg.prepare(scan.load_recent())
    out = [review(h, 7, False)]
    if monthly:
        out.append(review(h, 31, True))
    for title, body in out:
        print(title); print(body)
        if not dry:
            notify.send(title, body, color=15105570)
