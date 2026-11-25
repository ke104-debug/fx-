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
         "経済指標": "情報解析部隊", "ニュース": "情報解析部隊", "地政学": "情報解析部隊", "需給": "情報解析部隊", "政治財政": "情報解析部隊", "織り込み": "情報解析部隊"}


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
    # ドル主導か円主導か (EUR/USD・EUR/JPY)
    try:
        from . import strength
        stn = strength.now(h.Close)
        if stn:
            ops.append(dict(name="強さ|ドル主導の上げは続く(1時間)", bias=stn["bias"], reason=stn["text"]))
    except Exception as e:
        print("ドルと円の強さのエラー:", e)
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
    try:
        news = news + sp.headlines(query="為替介入 OR 財務官 OR 財務相", hours=48) + sp.headlines(query="首相 財政 OR 選挙", hours=48)
    except Exception:
        pass
    iv = an.intervention_now(news)
    ops.append(dict(name="政治財政|介入けん制の段階", bias=0.0, stage=iv["stage"], reason=f"段階{iv['stage']}: {iv['reason']}"))
    for name, fn in (("ニュース|見出しの向き", an.news_now), ("地政学|紛争・制裁のニュース", an.geo_now),
                     ("金融政策|日銀・FRBの見出し", an.policy_news_now), ("政治財政|財政・選挙の見出し", an.jp_politics_now)):
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
    # 需給の担当 (23年の検証では偏りが小さく、判断には使わない。今日の需給の出来事を伝えるだけ)
    try:
        from .. import quest
        j = now + pd.Timedelta(hours=9)
        ev = []
        jd = j.normalize().tz_localize(None)
        month = pd.date_range(jd.replace(day=1), jd + pd.offsets.MonthEnd(0), freq="D", tz="UTC") - pd.Timedelta(hours=9) + pd.Timedelta(hours=12)
        if jd in quest.gotobi_days(month):
            ev.append("五十日 (仲値9:55に向けたドル買い。2023年以降はほぼ効いていない)")
        ld = now.tz_convert("Europe/London")
        nxt = [ld.normalize() + pd.Timedelta(days=k) for k in range(1, 5)]
        if ld.weekday() < 5 and all(d.month != ld.month or d.weekday() >= 5 for d in nxt):
            ev.append("月末 (ロンドン16時のフィキシングで売買が集中。直後に戻りやすい傾向は弱い)")
        ny = now.tz_convert("America/New_York")
        if ny.weekday() < 5 and 8 <= ny.hour < 10:
            ev.append("NYカット (NY10時) 前。大口オプションの価格に吸い寄せられることがある")
        ops.append(dict(name="需給|今日の需給の出来事", bias=0.0, reason=" / ".join(ev) if ev else "目立つ需給の出来事なし"))
    except Exception:
        pass
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
    """方向の会議 (全統括AIの計算方法):
      1) 各担当の意見 b (-1〜+1) と重み w (検証部隊が23年・実際の記録で決めた値。0なら判断に入れない)
      2) 重複整理: 意見がよく似た担当のグループ (config の groups、相関0.6超) は、重みのある人数で重みを割る (グループで1票分)
      3) 部隊ごとの点数 = Σ(w×b) / Σw、全体 = 全担当の Σ(w×b) / Σw (重みの多い部隊ほど強く効く)
      4) 全体が +0.25 以上で買い寄り、-0.25 以下で売り寄り、それ以外は中立
      5) 意見が無い担当 (Gemini が答えない時など) は分子・分母の両方から外れる (黙った担当で結論が偏らない)
    その後 final() で、作戦部隊の合図・リスク管理部隊 (データ監視・資金管理を含む) の判定と合わせて最終結論を出す"""
    w = dict(weights() if w is None else w)
    try:
        grp = json.load(open(W_PATH)).get("groups", [])
    except Exception:
        grp = []
    present = {o["name"] for o in ops}
    for g in grp:
        live = [n for n in g if w.get(n, 0) > 0 and n in present]
        for n in live:
            w[n] = w[n] / len(live)
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


STALE_H = {"AI情報": 3, "ニュース": 26, "地政学": 26, "政治財政": 48}   # これより古い意見は使わない (時間)


def fresh(ops: list[dict], now=None) -> tuple[list[dict], list[str]]:
    """情報の鮮度: 古い意見を外す (as_of がある意見だけ判定)"""
    now = now or pd.Timestamp.now(tz="UTC")
    keep, dropped = [], []
    for o in ops:
        lim = STALE_H.get(o["name"].split("|")[0])
        if lim and o.get("as_of"):
            age = (now - pd.Timestamp(o["as_of"], tz="UTC")).total_seconds() / 3600
            if age > lim:
                dropped.append(f"{o['name'].split('|')[-1]} ({age:.0f}時間前の情報)")
                continue
        keep.append(o)
    return keep, dropped


def market_head(res: dict, pl: list[dict]) -> dict:
    """中間統括① 市場判断統括: 方向の会議 (情報解析・相場分析) + 作戦部隊の合図。
    意見が対立した時の決まり: 作戦部隊の合図 (23年で検証済みの作戦) を優先し、方向の会議が強く逆 (0.25以上) なら「対立」と記録する。
    対立した合図の成績は影の運用で採点し、悪ければ「対立なら見送り」に変える (検証してから)"""
    sent = [p for p in pl if str(p.get("sent")) == "True"]
    if sent:
        p = sent[-1]
        side = int(p["side"])
        conflict = res["overall"] * side <= -0.25
        agree = res["overall"] * side >= 0.1
        return {"side": side, "signal": p, "conflict": conflict, "agree": agree,
                "text": f"{'買い' if side == 1 else '売り'}の合図 ({p['entry']}・損切り{p['stop']})"
                        + ("。方向の会議も同じ向き" if agree else "。方向の会議と対立" if conflict else "")}
    return {"side": 0, "signal": None, "conflict": False, "agree": False,
            "text": {"買い寄り": "買い寄りだが作戦の合図はまだない", "売り寄り": "売り寄りだが作戦の合図はまだない",
                     "中立": "方向の材料が弱い"}[res["verdict"]]}


def risk_head(h: pd.DataFrame, ops: list[dict]) -> dict:
    """中間統括② リスク統括: リスク管理 + 資金管理 + データ監視 + 保有中 (決済・ポジション管理)"""
    from . import risk
    from .. import positions
    rk = risk.check(h, ops)
    held = positions.load()
    return {**rk, "held": held, "text": rk["level"] + (f" / 保有中 {len(held)}件" if held else "")}


def final(h: pd.DataFrame, ops: list[dict] | None = None) -> dict:
    """全統括AI (運用統括): 市場判断統括とリスク統括の結論だけを受け取って最終結論を出す。
    決まり: リスクが「警戒」なら新しく入らない > 作戦の合図があれば入る (対立・注意なら慎重) > 合図がなければ様子見"""
    import time as _t
    t0 = _t.time()
    ops = ops if ops is not None else opinions(h)
    ops, dropped = fresh(ops)
    res = decide(ops)
    pl = plans()
    mk, rh = market_head(res, pl), risk_head(h, ops)
    if rh["level"] == "警戒":
        concl = "新しく入らない (リスク統括が警戒)"
    elif mk["signal"] is not None:
        concl = mk["text"] + ("。リスクは注意" if rh["level"] == "注意" else "") + ("。慎重に" if mk["conflict"] or rh["level"] == "注意" else "")
    else:
        concl = "様子見 (" + mk["text"] + ")"
    res.update(risk=rh, plans=pl, conclusion=concl, ops=ops, market=mk, stale=dropped, secs=round(_t.time() - t0, 1))
    return res


def text(res: dict) -> str:
    L = [f"【全統括AI】{res.get('conclusion', res['verdict'])}",
         f"【方向の会議】{res['verdict']} (強さ {res['overall']:+.2f})" + ("　※24時間以内に重要指標" if res["caution"] else "")]
    if "market" in res:
        L.append(f"【市場判断統括】{res['market']['text']}")
    if "risk" in res:
        L.append(f"【リスク統括】{res['risk']['text']}" + "".join(f"\n  ・{who}: {why}" for who, why in res["risk"]["items"]))
    if res.get("stale"):
        L.append("【鮮度】古いので使わなかった意見: " + " / ".join(res["stale"]))
    try:
        from . import research
        rr = research.recent()
        if rr:
            L.append("【調査班の報告 (直近6時間)】" + "".join(f"\n  ・{x}" for x in rr))
    except Exception:
        pass
    for d, g in res["depts"].items():
        sc = "判断に使う意見なし" if g["score"] is None else f"{g['score']:+.2f}"
        L.append(f"【{d}】{sc}")
        for o in g["voters"]:
            L.append(f"  ・{o['name'].split('|')[1]} {o['bias']:+.2f} … {o['reason']}")
        pri = lambda o: (0 if o["name"].startswith("AI情報") else 1 if o["name"].split("|")[0] in ("ニュース", "地政学", "経済指標", "需給", "政治財政") else 2)
        for o in sorted([o for o in g["notes"] if o.get("reason") and o["reason"] != "合図あり"], key=pri)[:6]:
            L.append(f"  (参考) {o['name'].split('|')[1]} … {o['reason']}")
    return "\n".join(L)


def devil(res: dict) -> str:
    """Gemini に会議の結論への反対意見 (一番強い反論と、外れるならどんな時か) を出させる。監査AIの材料"""
    from . import groq
    r, who = groq.ask("反対意見", "あなたは為替 (ドル円) の取引チームの監査役。以下の会議の結論に、遠慮なく一番強い反論をする。"
                 "根拠は会議の中身とニュースのまとめだけ。JSON だけを返す: "
                 '{"counter": "反論60字以内", "risk": "結論が外れるとしたらどんな時か40字以内", "strength": 0〜2 の整数 (反論の強さ)}\n\n'
                 + text(res)[:2500])
    if not isinstance(r, dict):
        return ""
    return f"[{r.get('strength', '?')}] {str(r.get('counter', ''))[:80]} / 外れる時: {str(r.get('risk', ''))[:60]} ({who})"


ADVISED = ("作戦部隊", "情報解析部隊", "相場分析部隊", "市場判断統括", "リスク統括", "決済・ポジション管理部隊")
VIEW_UNITS = ("情報解析部隊", "相場分析部隊")    # 向きの見立ても出させて、後で当たりを採点する


def consult_gemini(res: dict) -> dict:
    """Gemini の相談役 (Groq の相談役とは別の目)。情報解析部隊と相場分析部隊の見立てだけを出させて記録し、Groq と比べる"""
    from . import gem
    r = gem.call("相談役G", "あなたは FX (ドル円) の売買判断チームの相談役。以下の会議の中身だけを根拠に、情報解析部隊と相場分析部隊それぞれの"
                 "24時間後のドル円の見立て (-1 下げ〜+1 上げ、分からなければ0) と、見落としを1つ (40字) 出す。"
                 'JSON だけ: {"情報解析部隊": {"見立て": 0, "見落とし": ""}, "相場分析部隊": {"見立て": 0, "見落とし": ""}}\n\n' + text(res)[:2800])
    out = {}
    if isinstance(r, dict):
        for u in ("情報解析部隊", "相場分析部隊"):
            v = r.get(u, {}) if isinstance(r.get(u), dict) else {}
            try:
                out[f"AI相談役G見立て|{u}"] = max(-1.0, min(1.0, float(v.get("見立て", 0))))
            except Exception:
                pass
            out[f"AI相談役G|{u}"] = str(v.get("見落とし", ""))[:60]
    return out


def consult(res: dict, every_hour=False) -> dict:
    """相談役AI (全部隊の補佐): 部隊ごとに「結論の弱いところ・見落とし・次に確かめること」を1つ助言。
    情報解析部隊と相場分析部隊の相談役は、自分の見立て (-1〜+1) も出す (記録して採点。判断には使わない)。
    Groq の無料枠がある時は毎時、無い時は Gemini で6時間に1回。1回の呼び出しで全部隊分"""
    from . import groq
    from .. import positions
    keys = ", ".join(f'"{u}": {{"助言": "", "調べてほしいこと": [""]' + (', "見立て": 0' if u in VIEW_UNITS else "") + "}" for u in ADVISED)
    r, who = groq.ask("相談役", "あなたは FX (ドル円) の売買判断チームの各部隊の相談役 (部隊ごとに1人ずつ)。以下の会議の中身を読み、"
                      "部隊ごとに「結論の弱いところ・見落とし・次に確かめるべきこと」を1つ、各50字以内で助言する。"
                      "情報解析部隊と相場分析部隊の相談役は、この後24時間のドル円の見立てを -1 (下げ)〜+1 (上げ) で付ける (分からなければ0)。"
                      "さらに、判断の精度を上げるために調査班に調べてほしいことを、ニュースを検索する言葉で最大2つ (例: \"日銀 利上げ 12月\")。"
                      "根拠は会議の中身と保有中だけ。JSON だけ: {" + keys + "}\n\n"
                      + text(res)[:2800] + f"\n【保有中】{positions.summary()}")
    if not isinstance(r, dict):
        return {}
    out = {}
    for u in ADVISED:
        v = r.get(u, {})
        if isinstance(v, str):
            v = {"助言": v}
        out[f"AI相談役|{u}"] = str(v.get("助言", ""))[:80]
        try:
            from . import research
            qs = v.get("調べてほしいこと") or []
            research.request(u, [x for x in (qs if isinstance(qs, list) else [qs]) if x])
        except Exception:
            pass
        if u in VIEW_UNITS:
            try:
                out[f"AI相談役見立て|{u}"] = max(-1.0, min(1.0, float(v.get("見立て", 0))))
            except Exception:
                pass
    out["AI相談役の担当"] = who
    return out


def _groq_ok(job):
    try:
        from . import groq
        return groq.allowed(job)
    except Exception:
        return False


def log(h: pd.DataFrame) -> dict:
    """影の運用: 会議の結果を journal/org.csv に1時間1行で残す (通知はしない)"""
    res = final(h)
    ops = res["ops"]
    # 30分ごとに1行 (毎時の分析は1時間に2回動く)。価格は最新の1分足に近い値を使う
    _now = pd.Timestamp.now(tz="UTC")
    key = f"{_now.floor('30min'):%Y-%m-%d %H:%M}"
    if os.path.exists(LOG) and key in set(pd.read_csv(LOG, usecols=["time_utc"]).time_utc):
        return res      # この30分はもう記録済み。AI を呼ばずに終わる (無料枠の節約)
    row = {"time_utc": key, "price": round(float(h.Close.iat[-1]), 3),
           "overall": res["overall"], "verdict": res["verdict"], "caution": res["caution"],
           "risk": res["risk"]["level"], "conclusion": res["conclusion"], "conflict": res["market"]["conflict"], "secs": res["secs"],
           "AI反対意見": devil(res) if (_now.hour % 2 == 0 and _now.minute < 30) else "",
           # 相談役: Groq の枠があれば毎時、無ければ Gemini で6時間に1回 (反対意見と重ならない 3・9・15・21時)
           **(consult(res) if (_groq_ok("相談役") or (_now.hour % 6 == 3 and _now.minute < 30)) else {}),
           **(consult_gemini(res) if (_now.hour % 2 == 1 and _now.minute < 30) else {}),
           "監査": "; ".join(__import__("fxpred.org.auditor", fromlist=["x"]).meeting_check(res))[:200],
           "介入けん制の段階": max([int(o.get("stage", 0)) for o in ops if "stage" in o] or [0]),
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
