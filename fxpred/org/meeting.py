"""会議: 専門AIの意見 → 部門の統括がまとめる → 統括AIが全体をまとめる。
重みは検証部 (score.py) が決めた config/org_weights.json。重み0の専門AIは「参考意見」として理由だけ載せる。

影の運用: 毎時の分析のたびに journal/org.csv に記録する (通知は変えない)。
python -m fxpred.org.meeting で今の会議の結果を表示する。"""
import json
import os

import numpy as np
import pandas as pd

from . import analysts as an, data as dt
from .. import specialists as sp

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
W_PATH = os.path.join(ROOT, "config", "org_weights.json")
LOG = os.path.join(ROOT, "journal", "org.csv")
# 大きな部隊: 全統括AI / 作戦部隊 / 情報解析部隊 / 相場分析部隊 / リスク管理部隊 / シミュレーション・検証部隊 / 改善研究部隊 + 監査AI (組織の外)
# 向きの意見を出すのは 情報解析部隊 と 相場分析部隊。作戦部隊は売買の合図、リスク管理部隊は取引してよい状況か、
# シミュレーション・検証部隊 (score.py・review.py) は重み、改善研究部隊 (advisor.py) は足りない部分の進言、監査AI (auditor.py) は問題点
DEPTS = {"相場分析": "相場分析部隊", "強さ": "相場分析部隊", "テクニカル": "相場分析部隊", "ローソク足": "相場分析部隊",
         "チャートパターン": "相場分析部隊", "サポレジ": "相場分析部隊", "マルチタイム": "相場分析部隊", "ボラティリティ": "相場分析部隊",
         "市場構造": "相場分析部隊", "AI情報": "情報解析部隊", "金融政策": "情報解析部隊", "他市場": "情報解析部隊", "センチメント": "情報解析部隊",
         "経済指標": "情報解析部隊", "ニュース": "情報解析部隊", "地政学": "情報解析部隊"}


def weights() -> dict:
    return json.load(open(W_PATH)).get("weights", {}) if os.path.exists(W_PATH) else {}


def opinions(h: pd.DataFrame, now=None) -> list[dict]:
    """今の時点の専門AI全員の意見。h は1時間足 (確定足)"""
    now = now or pd.Timestamp.now(tz="UTC")
    uj = dt.usdjpy_jst10(h)
    today = (now + pd.Timedelta(hours=9)).normalize().tz_localize(None)
    days = uj.index.union([today])
    uj = uj.reindex(days).ffill()
    x = dt.at_jst10(dt.fetch(start=f"{today.year - 2}-01-01"), days)
    b = an.features(x, uj, None).iloc[-1]
    ops = [dict(name=k, bias=float(v) if pd.notna(v) else 0.0, reason=_reason(k, x)) for k, v in b.items()]
    # 相場分析部 (今の確定足から)
    c = h.Close
    hi, lo = h.High.rolling(120).max().iat[-1], h.Low.rolling(120).min().iat[-1]
    pos = (c.iat[-1] - lo) / (hi - lo) if hi > lo else 0.5
    d = c.diff(); up, dn = d.clip(lower=0).ewm(alpha=1 / 14).mean(), (-d.clip(upper=0)).ewm(alpha=1 / 14).mean()
    rsi = float((100 - 100 / (1 + up / dn)).iat[-1])
    rg = 1 if pos <= .25 and rsi < 45 else -1 if pos >= .75 and rsi > 55 else 0
    ops.append(dict(name="相場分析|レンジの端で逆張り", bias=float(rg),
                    reason=f"直近120時間の中の位置 {pos:.0%}・RSI {rsi:.0f}"))
    e50, e200 = c.ewm(span=50).mean().iat[-1], c.ewm(span=200).mean().iat[-1]
    ops.append(dict(name="相場分析|トレンド(50/200本)", bias=float(np.sign(e50 - e200)), reason="50本平均が200本平均の" + ("上" if e50 > e200 else "下")))
    ops.append(dict(name="相場分析|前日からの勢い", bias=float(np.sign(c.iat[-1] - c.iat[-25])), reason=f"24時間で {(c.iat[-1] - c.iat[-25]) * 100:+.0f}pips"))
    # ⑩〜⑮ 相場分析部の専門AI (1時間足の最新の確定足)
    try:
        from . import chart
        cb = chart.features(h).iloc[-1]
        for k, v in cb.items():
            ops.append(dict(name=k, bias=float(v) if pd.notna(v) else 0.0, reason="合図あり" if v else ""))
    except Exception as e:
        print("相場分析部のエラー:", e)
    # 履歴で検証できない専門AI (記録を始めた日から採点)
    news = sp.headlines(hours=24)
    for name, fn in (("ニュース|見出しの向き", an.news_now), ("地政学|紛争・制裁のニュース", an.geo_now),
                     ("金融政策|日銀・FRBの見出し", an.policy_news_now)):
        try:
            o = fn(news)
            ops.append(dict(name=name, bias=o["bias"], reason=o["reason"]))
        except Exception as e:
            ops.append(dict(name=name, bias=0.0, reason=f"取得できず ({e})"))
    try:   # Gemini (無料枠) に見出しを読ませる専門AI。キーが無ければ何もしない
        from . import llm
        ops += llm.opinions(float(h.Close.iat[-1]))
    except Exception as e:
        print("AI情報のエラー:", e)
    o = an.events_now(now)
    ops.append(dict(name="経済指標|重要指標の予定", bias=0.0, reason=o["reason"], caution=o.get("caution", False)))
    return ops


def _reason(k: str, x: pd.DataFrame) -> str:
    last = x.iloc[-1]
    def ch(col, n, log=True):
        s = x[col].dropna()
        if len(s) <= n:
            return np.nan
        return (np.log(s.iat[-1] / s.iat[-1 - n]) * 100) if log else (s.iat[-1] - s.iat[-1 - n]) * 100
    txt = {
        "金融政策|米2年金利の勢い(5日)": lambda: f"米2年金利 {last.get('米2年', np.nan):.2f}% (5日で {ch('米2年', 5, False):+.0f}bp)",
        "金融政策|米2年金利の流れ(20日)": lambda: f"米2年金利 20日で {ch('米2年', 20, False):+.0f}bp",
        "金融政策|政策金利の方向(60日)": lambda: f"FF金利 {last.get('FF金利', np.nan):.2f}%",
        "金融政策|日米金利差の変化(20日)": lambda: f"日米10年金利差 {last.get('米10年', np.nan) - last.get('日10年', np.nan):.2f}%",
        "他市場|米株の勢い(5日)": lambda: f"S&P500 5日で {ch('米株', 5):+.1f}%",
        "他市場|日経の勢い(5日)": lambda: f"日経平均 5日で {ch('日経', 5):+.1f}%",
        "他市場|金の勢い(5日)・逆": lambda: f"金 5日で {ch('金', 5):+.1f}%",
        "他市場|原油の勢い(5日)": lambda: f"原油 5日で {ch('原油', 5):+.1f}%",
        "他市場|ドル指数の勢い(5日)": lambda: f"ドル指数 {last.get('ドル指数', np.nan):.2f} (5日で {ch('ドル指数', 5):+.1f}%)",
        "他市場|米10年金利(前日)": lambda: f"米10年金利 {last.get('米10年(Yahoo)', np.nan):.2f}%",
        "センチメント|VIXの高さ・逆": lambda: f"VIX {last.get('VIX', np.nan):.1f}",
        "センチメント|VIXの上がり方(5日)・逆": lambda: f"VIX 5日で {ch('VIX', 5, False) / 100:+.1f}",
    }.get(k)
    try:
        return txt() if txt else ""
    except Exception:
        return ""


def decide(ops: list[dict], w: dict | None = None) -> dict:
    """部門ごとに重み付き平均 → 統括。戻り値: 全体の向き・確からしさ・部門の結論・参考意見"""
    w = weights() if w is None else w
    dept = {}
    for o in ops:
        sec = o["name"].split("|")[0]
        g = dept.setdefault(DEPTS.get(sec, sec), {"sum": 0.0, "w": 0.0, "voters": [], "notes": []})
        wt = w.get(o["name"], 0.0)
        if wt > 0:
            g["sum"] += wt * o["bias"]; g["w"] += wt; g["voters"].append(o)
        elif o.get("reason"):
            g["notes"].append(o)
    tot_w = sum(g["w"] for g in dept.values())
    overall = sum(g["sum"] for g in dept.values()) / tot_w if tot_w else 0.0
    for g in dept.values():
        g["score"] = g["sum"] / g["w"] if g["w"] else None
    caution = any(o.get("caution") for o in ops)
    verdict = "買い寄り" if overall >= 0.25 else "売り寄り" if overall <= -0.25 else "中立"
    return {"overall": round(overall, 3), "verdict": verdict, "depts": dept, "caution": caution}


def plans(hours=3) -> list[dict]:
    """作戦部隊: 直近 hours 時間に出た売買の合図 (送ったもの・見送ったもの)"""
    p = os.path.join(ROOT, "journal", "signals.csv")
    if not os.path.exists(p):
        return []
    s = pd.read_csv(p)
    s["t"] = pd.to_datetime(s.time_utc, utc=True)
    s = s[s.t >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)]
    return s.to_dict("records")


def final(h: pd.DataFrame, ops: list[dict] | None = None) -> dict:
    """全統括AI: 方向の会議 + 作戦部隊の合図 + リスク管理部隊の判定 → 最終結論"""
    from . import risk
    ops = ops if ops is not None else opinions(h)
    res = decide(ops)
    rk = risk.check(h, ops)
    pl = plans()
    sent = [p for p in pl if str(p.get("sent")) == "True"]
    if rk["level"] == "警戒":
        concl = "新しく入らない (リスク管理部隊が警戒)"
    elif sent:
        p = sent[-1]
        side = "買い" if int(p["side"]) == 1 else "売り"
        agree = (res["overall"] > 0.1 and int(p["side"]) == 1) or (res["overall"] < -0.1 and int(p["side"]) == -1)
        against = (res["overall"] < -0.1 and int(p["side"]) == 1) or (res["overall"] > 0.1 and int(p["side"]) == -1)
        concl = f"{side}の合図あり ({p['entry']}・損切り{p['stop']})" + ("。方向の会議も同じ向き" if agree else "。方向の会議は逆向き、慎重に" if against else "")
    else:
        concl = {"買い寄り": "買い寄りだが作戦の合図はまだない (下値の押し目を待つ)", "売り寄り": "売り寄りだが作戦の合図はまだない (戻りを待つ)",
                 "中立": "様子見 (方向の材料が弱い)"}[res["verdict"]]
    res.update(risk=rk, plans=pl, conclusion=concl, ops=ops)
    return res


def text(res: dict) -> str:
    L = [f"【全統括AI】{res.get('conclusion', res['verdict'])}",
         f"【方向の会議】{res['verdict']} (強さ {res['overall']:+.2f})" + ("　※24時間以内に重要指標" if res["caution"] else "")]
    if "risk" in res:
        L.append(f"【リスク管理部隊】{res['risk']['level']}" + "".join(f"\n  ・{who}: {why}" for who, why in res["risk"]["items"]))
    for d, g in res["depts"].items():
        sc = "判断に使う意見なし" if g["score"] is None else f"{g['score']:+.2f}"
        L.append(f"【{d}】{sc}")
        for o in g["voters"]:
            L.append(f"  ・{o['name'].split('|')[1]} {o['bias']:+.2f} … {o['reason']}")
        pri = lambda o: (0 if o["name"].startswith("AI情報") else 1 if o["name"].split("|")[0] in ("ニュース", "地政学", "経済指標") else 2)
        for o in sorted([o for o in g["notes"] if o.get("reason") and o["reason"] != "合図あり"], key=pri)[:6]:
            L.append(f"  (参考) {o['name'].split('|')[1]} … {o['reason']}")
    return "\n".join(L)


def log(h: pd.DataFrame) -> dict:
    """影の運用: 会議の結果を journal/org.csv に1時間1行で残す (通知はしない)"""
    res = final(h)
    ops = res["ops"]
    row = {"time_utc": f"{h.index[-1] + pd.Timedelta(hours=1):%Y-%m-%d %H:%M}", "price": round(float(h.Close.iat[-1]), 3),
           "overall": res["overall"], "verdict": res["verdict"], "caution": res["caution"],
           "risk": res["risk"]["level"], "conclusion": res["conclusion"],
           **{o["name"]: round(o["bias"], 3) for o in ops}}
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    if os.path.exists(LOG):
        old = pd.read_csv(LOG)
        if row["time_utc"] in set(old.time_utc):
            return res
        pd.concat([old, pd.DataFrame([row])]).to_csv(LOG, index=False)
    else:
        pd.DataFrame([row]).to_csv(LOG, index=False)
    return res


if __name__ == "__main__":
    from .. import scan
    h = scan.load_recent()
    print(text(final(h)))
