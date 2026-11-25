"""朝の反省会: 前の日 (日本時間10時〜今朝10時) を振り返り、何が良かったか・悪かったかをはっきり言う。
システムの改善 (ルールや重みの変更) はここではしない (凍結中)。記録は journal/reflections.csv。
見るもの: 値動き / 朝の予想 (値幅・下値・上値) / 送った合図と見送った合図のその後 / 組織の会議の向き / 影の運用 / 指標の実況シミュレーション / 止まった処理"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
J = os.path.join(ROOT, "journal")


def _read(name):
    p = os.path.join(J, name)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def run(h: pd.DataFrame, now=None) -> dict:
    now = now or pd.Timestamp.now(tz="UTC")
    jst = now + pd.Timedelta(hours=9)
    t1 = (jst.normalize() + pd.Timedelta(hours=10)) - pd.Timedelta(hours=9)     # 今朝10時 (UTC)
    t0 = t1 - pd.Timedelta(days=1)
    w = h[(h.index >= t0) & (h.index < t1)]
    good, bad, facts = [], [], []
    if len(w):
        mv = (w.Close.iat[-1] - w.Open.iat[0]) * 100
        facts.append(f"値動き: {w.Open.iat[0]:.2f} → {w.Close.iat[-1]:.2f} ({mv:+.0f}pips)、高値 {w.High.max():.2f} / 安値 {w.Low.min():.2f} (値幅 {(w.High.max() - w.Low.min()) * 100:.0f}pips)")
    # 朝の予想
    fc = _read("daily_forecast.csv")
    if len(fc) and len(w):
        fc["t"] = pd.to_datetime(fc.time_utc, utc=True)
        f = fc[(fc.t >= t0 - pd.Timedelta(hours=2)) & (fc.t < t1)]
        if len(f):
            f = f.iloc[-1]
            hi, lo = w.High.max(), w.Low.min()
            in_rng = lo >= f.exp_low - 0.05 and hi <= f.exp_high + 0.05
            exp_w, act_w = (f.exp_high - f.exp_low) * 100, (hi - lo) * 100
            (good if abs(act_w - exp_w) <= 0.3 * exp_w else bad).append(
                f"値幅の予想 {exp_w:.0f}pips → 実際 {act_w:.0f}pips" + (" (予想の範囲に収まった)" if in_rng else " (予想の範囲を外れた)"))
            for name, lv, broke in (("下値", f.support, lo < f.support - 0.05), ("上値", f.resistance, hi > f.resistance + 0.05)):
                facts.append(f"{name} {lv:.2f}: " + ("抜けた" if broke else "守られた"))
    # 送った合図・見送った合図
    try:
        from ..periodic import evaluate_signals
        ev = evaluate_signals(h)
        if len(ev):
            ev["t"] = pd.to_datetime(ev.time_utc, utc=True)
            d = ev[(ev.t >= t0) & (ev.t < t1)]
            s, k = d[d.sent.astype(str) == "True"], d[d.sent.astype(str) == "False"]
            if len(s):
                p = s.pips.sum()
                (good if p > 0 else bad).append(f"送った合図 {len(s)}回: 合計 {p:+.0f}pips (" + " / ".join(f"{r.strategy} {r.result}" for r in s.itertuples()) + ")")
            else:
                facts.append("送った合図: なし")
            if len(k):   # 3時間以内に同じ向きで何度も出た合図は1回として数える (同じ取引の重複)
                k = k.sort_values("t")
                keep, last = [], {}
                for r in k.itertuples():
                    if r.side in last and r.t - last[r.side] < pd.Timedelta(hours=3):
                        continue
                    last[r.side] = r.t
                    keep.append(r.Index)
                k = k.loc[keep]
                kp = k.pips.dropna()
                if len(kp):
                    (bad if kp.sum() > 10 else good).append(
                        f"見送った合図 {len(k)}回: 入っていたら合計 {kp.sum():+.0f}pips" + (" → 見送りでチャンスを逃した" if kp.sum() > 10 else " → 見送りは正しかった"))
    except Exception as e:
        facts.append(f"合図の答え合わせができず ({e})")
    # 組織の会議の向き (その時から4時間後の動きと比べる)
    og = _read("org.csv")
    if len(og):
        og["t"] = pd.to_datetime(og.time_utc, utc=True)
        o = og[(og.t >= t0) & (og.t < t1 - pd.Timedelta(hours=4))]
        if len(o):
            c = h.Close
            fut = [c[c.index >= t + pd.Timedelta(hours=3)].iloc[0] - p if (c.index >= t + pd.Timedelta(hours=3)).any() else np.nan
                   for t, p in zip(o.t, o.price)]
            o = o.assign(fut=fut)
            strong = o[o.overall.abs() >= 0.25]
            if len(strong):
                hit = (np.sign(strong.overall) == np.sign(strong.fut)).mean()
                (good if hit >= 0.5 else bad).append(f"組織の会議: 向きを出した{len(strong)}回のうち、4時間後に当たったのは {hit:.0%}")
            else:
                facts.append("組織の会議: 一日中ほぼ中立 (向きを出さなかった)")
            if "AI反対意見" in o:
                n = o["AI反対意見"].astype(str).str.startswith("[2]").sum()
                if n:
                    facts.append(f"Gemini の強い反対意見 {n}回")
    # 影の運用
    try:
        from .. import lab, shadow
        d = lab.prep(h.iloc[-1200:])
        for name, c in shadow.CANDIDATES.items():
            t = c["fn"](d, **c["params"])
            t = t[(t.time >= t0) & (t.time < t1)] if len(t) else t
            if len(t):
                pp = t.pips if "pips" in t else t.R * t.risk
                facts.append(f"影の運用「{name[:16]}」: {len(t)}回 {pp.sum():+.0f}pips")
    except Exception:
        pass
    # 指標の実況シミュレーション
    es = _read("eventsim.csv")
    if len(es):
        es["t"] = pd.to_datetime(es.release_utc, utc=True)
        e = es[(es.t >= t0) & (es.t < t1)]
        for r in e.itertuples():
            facts.append(f"指標シミュレーション {str(r.events)[:30]}: 順張り {r.A if r.A == r.A else '入らず'} / 逆張り {r.B if r.B == r.B else '入らず'} / 組織 {r.C:+.1f}pips")
    if not good:
        good.append("目立って良かった点はなし")
    if not bad:
        bad.append("目立って悪かった点はなし")
    res = {"date": f"{(t0 + pd.Timedelta(hours=9)):%Y-%m-%d}", "facts": facts, "good": good, "bad": bad}
    # 朝の通知は軽くする (利用者の指示)。詳しい反省は記録に残し、聞かれた時と週次の見直しで使う。Gemini は呼ばない
    try:
        p = os.path.join(J, "reflections.csv")
        row = {"date": res["date"], "facts": " | ".join(facts), "good": " | ".join(good), "bad": " | ".join(bad), "ai": res.get("ai", "")}
        old = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
        if not len(old) or res["date"] not in set(old.date.astype(str)):
            pd.concat([old, pd.DataFrame([row])]).to_csv(p, index=False)
    except Exception:
        pass
    return res


def lines(res: dict) -> list[str]:
    """朝の通知用に2行だけ (詳しくは journal/reflections.csv)"""
    return ["━━━━━━━━━━", f"昨日の反省: ○ {res['good'][0][:45]} / × {res['bad'][0][:45]}"]
