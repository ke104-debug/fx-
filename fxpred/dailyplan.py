"""毎日の予想・振り返り・自己改善 (毎朝10時 日本時間に実行する想定。追加料金なし)

  予想     : 今日の方向 (買い寄り/売り寄り/中立)、予想値幅、下値・上値、主担当、各担当の意見を記録
  振り返り : 24時間後に、実際の値動き・値幅・下値/上値の攻防と予想を比べ、当たり外れと原因を記録
  自己改善 : 直近の成績から、担当AIごとの重み (当たる担当の意見を重く) と予想値幅の補正を自動で調整する
            (journal/learning.json)。調整幅には上限を設け、少ない成績で極端に変えない
"""
import json
import os
from datetime import timedelta

import numpy as np
import pandas as pd

from . import edge, regime as rg, specialists as sp

J = os.path.join(os.path.dirname(__file__), "..", "journal")
FORECAST = os.path.join(J, "daily_forecast.csv")
REVIEW = os.path.join(J, "daily_review.csv")
LEARN = os.path.join(J, "learning.json")
JST = "Asia/Tokyo"


def load_learning() -> dict:
    return json.load(open(LEARN)) if os.path.exists(LEARN) else {"weights": {}, "range_scale": 1.0, "notes": []}


def analyze(h: pd.DataFrame, learn: dict) -> dict:
    reg = rg.NAMES[h.regime.iat[-1]]
    anom = edge.anomaly(h)
    news = sp.headlines()
    views = {"急変担当": sp.shock_view(h, anom, news), "レンジ担当": sp.range_view(h, h.rsi.iat[-1]),
             "トレンド担当": sp.trend_view(h)}
    coord = sp.coordinate(reg, views, learn)
    return {"regime": reg, "anom": anom, "views": views, "coord": coord, "news": news}


def forecast(h: pd.DataFrame, a: dict, learn: dict) -> dict:
    c = float(h.Close.iat[-1])
    # 直近5日の値幅 (24時間ごと) の中央値をもとにする
    day_rng = float((h.High.rolling(24).max() - h.Low.rolling(24).min()).iloc[-120::24].median())
    rng = day_rng * learn.get("range_scale", 1.0)
    shift = {"買い寄り": 0.2, "売り寄り": -0.2, "中立": 0.0}[a["coord"]["verdict"]] * rng
    rv = a["views"]["レンジ担当"]
    row = {"date": f"{(h.index[-1] + timedelta(hours=1)).tz_convert(JST):%Y-%m-%d}", "time_utc": f"{h.index[-1] + timedelta(hours=1):%Y-%m-%d %H:%M}",
           "price": round(c, 3), "regime": a["regime"], "lead": a["coord"]["lead"], "verdict": a["coord"]["verdict"],
           "score": a["coord"]["score"], "exp_high": round(c + rng / 2 + shift, 2), "exp_low": round(c - rng / 2 + shift, 2),
           "support": rv["support"], "resistance": rv["resistance"],
           **{f"bias_{k}": v["bias"] for k, v in a["views"].items()}}
    os.makedirs(J, exist_ok=True)
    if os.path.exists(FORECAST) and row["time_utc"] in set(pd.read_csv(FORECAST).time_utc):
        return row   # 週末などで同じ足からの予想は1回だけ記録する
    pd.DataFrame([row]).to_csv(FORECAST, mode="a", header=not os.path.exists(FORECAST), index=False)
    return row


def review(h: pd.DataFrame, learn: dict) -> list[dict]:
    """24時間たった予想を答え合わせする (まだのものだけ)"""
    if not os.path.exists(FORECAST):
        return []
    f = pd.read_csv(FORECAST)
    done = set(pd.read_csv(REVIEW).time_utc) if os.path.exists(REVIEW) else set()
    out = []
    for _, r in f.iterrows():
        t0 = pd.Timestamp(r.time_utc, tz="UTC")
        if r.time_utc in done:
            continue
        w = h[h.index >= t0].iloc[:24]   # 予想後の24本 (取引のある24時間。週末をまたぐ場合も)
        if len(w) < 24:
            continue
        hi, lo, close = w.High.max(), w.Low.min(), w.Close.iat[-1]
        move = close - r.price
        actual = "上昇" if move > 0.15 else "下落" if move < -0.15 else "横ばい"
        want = {"買い寄り": "上昇", "売り寄り": "下落", "中立": "横ばい"}[r.verdict]
        # 中立の予想は「大きく動かなかった (0.3円以内)」なら正解とする
        hit_dir = abs(move) <= 0.3 if want == "横ばい" else actual == want
        rng_ratio = (hi - lo) / max(r.exp_high - r.exp_low, 0.05)
        in_range = hi <= r.exp_high + 0.1 and lo >= r.exp_low - 0.1
        sup_broken, res_broken = lo < r.support - 0.1, hi > r.resistance + 0.1
        spec = {}
        for name in ("急変担当", "レンジ担当", "トレンド担当"):
            b = r.get(f"bias_{name}", 0)
            spec[name] = None if b == 0 else ((b > 0) == (move > 0) and abs(move) > 0.15)
        lessons = []
        if not hit_dir:
            lessons.append(f"方向を外した: 予想 {r.verdict} → 実際 {actual} ({move:+.2f}円)")
        if rng_ratio > 1.3:
            lessons.append(f"値幅を小さく見積もった (実際は予想の{rng_ratio:.1f}倍)")
        elif rng_ratio < 0.6:
            lessons.append(f"値幅を大きく見積もった (実際は予想の{rng_ratio:.1f}倍)")
        if sup_broken: lessons.append(f"下値 {r.support} を割り込んだ")
        if res_broken: lessons.append(f"上値 {r.resistance} を上抜けた")
        wrong = [k for k, v in spec.items() if v is False]
        if wrong: lessons.append("外した担当: " + "、".join(wrong))
        # 大きくずれたかどうか (小さなずれは「想定内」として反省しない)
        big = []
        # 基準: 方向を外して0.8円以上動いた / 値幅が予想の2倍以上か0.4倍以下 / 下値・上値を0.5円以上抜けて、その外で終わった
        if not hit_dir and abs(move) >= 0.8: big.append("方向")
        if rng_ratio >= 2.5 or rng_ratio <= 0.35: big.append("値幅")
        if (lo < r.support - 0.5 and close < r.support) or (hi > r.resistance + 0.5 and close > r.resistance): big.append("価格帯")
        cause = ""
        if big:
            sd = h.Close.diff().rolling(480).std().reindex(w.index).iat[0]
            zmax = (w.Close.diff().abs() / sd).max()
            cause = f"その日の最大の1時間の動きは普段の{zmax:.1f}倍"
            try:
                c = sp.classify(sp.headlines(hours=36))
                if c: cause += "、ニュースの主な話題: " + "、".join(f"{k}({n}件)" for k, n, _ in c[:2])
            except Exception:
                pass
        row = {"time_utc": r.time_utc, "date": r.date, "verdict": r.verdict, "actual": actual, "move": round(move, 2),
               "hit_dir": hit_dir, "high": round(hi, 2), "low": round(lo, 2), "range_ratio": round(rng_ratio, 2),
               "in_range": in_range, "support_broken": sup_broken, "resistance_broken": res_broken,
               **{f"hit_{k}": v for k, v in spec.items()}, "big_miss": "・".join(big), "cause": cause,
               "lessons": " / ".join(lessons) or "予想どおり"}
        pd.DataFrame([row]).to_csv(REVIEW, mode="a", header=not os.path.exists(REVIEW), index=False)
        out.append(row)
    if out:
        improve(learn)
    return out


def improve(learn: dict) -> dict:
    """反省と改善。小さなずれでは何も変えない。
    同じ種類の大きなずれが直近20日で3回以上くり返された時だけ、その部分を少しだけ調整する (偶然で変えないため)"""
    rv = pd.read_csv(REVIEW).tail(20)
    notes = []
    big = rv.get("big_miss", pd.Series(dtype=str)).fillna("")
    n_dir, n_rng = big.str.contains("方向").sum(), big.str.contains("値幅").sum()
    if n_dir >= 3:
        for name in ("急変担当", "レンジ担当", "トレンド担当"):
            col = rv.get(f"hit_{name}")
            calls = col.dropna().astype(str).map({"True": 1, "False": 0}).dropna() if col is not None else []
            if len(calls) >= 5:
                acc = calls.mean()
                wgt = float(np.clip(0.5 + (acc - 0.5) * 2, 0.5, 1.5))
                old = learn["weights"].get(name, 1.0)
                learn["weights"][name] = round(0.7 * old + 0.3 * wgt, 2)
                if abs(learn["weights"][name] - old) >= 0.01:
                    notes.append(f"{name}の重み {old} → {learn['weights'][name]} (直近{len(calls)}回の正解率 {acc:.0%})")
        if not notes:
            notes.append(f"方向の大外れが{n_dir}回。担当ごとの成績に差がないため重みは変えず、見送り基準を維持")
    if n_rng >= 3:
        rr = rv[big.str.contains("値幅")].range_ratio
        wide, narrow = (rr >= 2.5).sum(), (rr <= 0.35).sum()
        old = learn.get("range_scale", 1.0)
        if wide >= 0.75 * len(rr):
            learn["range_scale"] = round(min(1.5, old * 1.15), 3)
            notes.append(f"値幅を小さく見積もる大外れが{wide}回 → 予想値幅を x{old} から x{learn['range_scale']} に")
        elif narrow >= 0.75 * len(rr):
            learn["range_scale"] = round(max(0.7, old * 0.85), 3)
            notes.append(f"値幅を大きく見積もる大外れが{narrow}回 → 予想値幅を x{old} から x{learn['range_scale']} に")
        else:
            notes.append(f"値幅の大外れ{len(rr)}回は広すぎ・狭すぎが混在 (相場の荒れ方の変化) → 変更しない")
    if not notes:
        notes.append("大きなずれのくり返しはなし → 変更しない (偶然の外れで設定を変えない)")
    learn["notes"] = notes
    learn["updated"] = f"{pd.Timestamp.now(tz=JST):%Y-%m-%d %H:%M}"
    json.dump(learn, open(LEARN, "w"), ensure_ascii=False, indent=1)
    return learn


def run_health() -> str | None:
    """直近24時間の自動実行の回数 (GitHub の定期実行が間引かれていないかを毎朝確かめる)"""
    import urllib.request
    repo, token = os.getenv("GITHUB_REPOSITORY"), os.getenv("GITHUB_TOKEN")
    if not repo or not token:
        return None
    since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = []
    for wf, label, expect in (("fx-quick.yml", "5分監視", 288), ("fx-auto.yml", "毎時の分析", 48)):
        # 定期実行 (schedule) と Cloudflare の時計役 (workflow_dispatch) の両方を数える。成功した回だけ
        url = f"https://api.github.com/repos/{repo}/actions/workflows/{wf}/runs?status=success&created=>={since}&per_page=1"
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
            n = json.load(urllib.request.urlopen(req, timeout=15)).get("total_count", 0)
        except Exception:
            continue
        mark = "✅" if n >= expect * 0.8 else "⚠️" if n >= expect * 0.3 else "❌"
        out.append(f"{mark} {label} {n}/{expect}回")
    return "稼働 (直近24時間): " + " / ".join(out) if out else None


def morning_message(h: pd.DataFrame, a: dict, fc: dict, reviews: list[dict], learn: dict) -> tuple[str, str]:
    """朝の通知 (視認性重視): 今日の判断と売買ライン → 昨日の振り返り の順"""
    from . import fmt
    v = a["views"]
    sup, res, c = fc["support"], fc["resistance"], fc["price"]
    lines = []
    # 朝の方向判断 (買い寄り/売り寄り) は検証で正解率が低かった (125日中、方向を言った16日で25%) ため出さない。
    # 代わりに、買う場合・売る場合それぞれの位置と損切り・利確を示し、実際のチャンスは監視が見つけて通知する
    plan = None
    lines += ["**判断　　: 様子見で待つ**（方向の予想は精度が低いため出しません。チャンスは監視が通知）",
              f"**買うなら: {sup:.2f} 付近で反発確認 / 損切り {sup - 0.20:.2f} / 利確 {res - 0.10:.2f}**",
              f"**売るなら: {res:.2f} 付近で反落確認 / 損切り {res + 0.20:.2f} / 利確 {sup + 0.10:.2f}**",
              f"担当AIの見立て (参考): {fc['verdict']}"]
    # 信頼度は売買シグナルと同じ計算 (条件の一致度)。様子見の日は「買い・売りどちらも条件がそろっていない」ことを示す
    from . import edge, scan
    try:
        ctx = scan.context(h)
        reg = a["regime"]
        best = 0
        for sd in (1, -1):
            if plan and sd != (1 if plan[0] == "買い" else -1):
                continue
            n_ok, ok, ng = edge.confluence(sd, ctx["tf"], scan.near_level(h, sd), ctx["p_up"], ctx["vix_chg"], reg, False)
            agree = 1 if fc["verdict"] == ("買い寄り" if sd == 1 else "売り寄り") else 0
            best = max(best, edge.confidence(ok, agree, 0, learn.get("cond_weights")))
        conf = best
    except Exception:
        conf = 0
    lines += [(f"**信頼度　: {conf}%**　" if plan else "") + f"予想値幅 {fc['exp_low']:.2f}〜{fc['exp_high']:.2f}",
              f"現在 {c:.2f}　下値 {sup:.2f} / 上値 {res:.2f}　相場: {a['regime']}"]
    try:
        from . import calendar as cal
        evs = cal.upcoming(pd.Timestamp.now(tz="UTC"), 24, ("High", "Medium"))
    except Exception:
        evs = []
    if evs:
        lines += ["今日の指標 (Forex Factory: 🔴重要 🟠中)"] + ["・" + cal.fmt(e) for e in evs[:6]]
    else:
        lines.append("今日の指標: 🔴🟠 の発表なし")
    lines += ["━━━━━━━━━━", "担当AIの見立て",
              f"・トレンド: {v['トレンド担当'].get('direction', '')} / {v['トレンド担当'].get('status', '')}",
              f"・レンジ: {v['レンジ担当']['lines'][-1]}",
              f"・急変: " + (v['急変担当']['causes'][0][0] + "の話題が多い" if v['急変担当'].get('causes') else "目立った材料なし")]
    if reviews:
        r = reviews[-1]
        allr = pd.read_csv(REVIEW)
        lines += ["━━━━━━━━━━", "昨日の振り返り",
                  f"{'○' if r['hit_dir'] else '×'} 予想 {r['verdict']} → 実際 {r['move']:+.2f}円（{r['low']}〜{r['high']}）"]
        if r.get("big_miss"):
            lines += [f"大きなずれ: {r['big_miss']}", f"原因の推定: {r['cause'] or '不明'}",
                      "改善: " + " / ".join(learn.get("notes", [])[:2])]
        else:
            lines.append("ずれは想定内（変更なし）")
        lines.append(f"通算: 方向 {allr.hit_dir.mean():.0%}（{len(allr)}日）")
    try:   # AI の組織 (検証中): 全統括AIの結論・リスク管理部隊・Gemini のまとめ・監査AI
        from .org import meeting, auditor
        org_res = meeting.final(h)
        lines += ["━━━━━━━━━━", "AIの組織の会議 (検証中・参考)", f"・全統括AI: {org_res['conclusion']}",
                  f"・リスク管理部隊: {org_res['risk']['level']}" + (f" ({org_res['risk']['items'][0][1]})" if org_res["risk"]["items"] else "")]
        summ = [o["reason"] for o in org_res["ops"] if o["name"] == "AI情報|まとめ"]
        if summ:
            lines.append(f"・情報解析部隊 (AI): {summ[0]}")
        found = auditor.audit(h, res=org_res)
        lines.append(f"・監査AI: {found[0]}")
        lines += [f"・監査の補助: {x}" for x in auditor.ai_audit(found, org_res)[:2]]
        try:   # 相談役AI の最新の助言 (3時間に1回の記録から)
            og = pd.read_csv(os.path.join(J, "org.csv"))
            cols = [c for c in og.columns if c.startswith("AI相談役|")]
            if cols:
                last = og[cols].dropna(how="all").tail(1)
                for c in cols:
                    v = str(last[c].iat[0]) if len(last) else ""
                    if v and v != "nan":
                        lines.append(f"・相談役 ({c.split('|')[1]}): {v[:60]}")
        except Exception:
            pass
        from .org import gem
        ai = gem.call("朝の見立て", "ドル円の今日の見立てを、以下の会議の内容と指標の予定だけから日本語で3行 (各40字以内) で書く。"
                      "1行目: 今日の相場の見どころ、2行目: 気をつける時間と材料、3行目: 入るならどんな形か (入らない方がよければそう書く)。"
                      "数字は会議にあるものだけ使う。JSON だけ: {\"lines\": [\"\", \"\", \"\"]}\n\n" + meeting.text(org_res)[:2500])
        if isinstance(ai, dict) and ai.get("lines"):
            lines += ["・AIの見立て (Gemini):"] + [f"　{str(x)[:60]}" for x in ai["lines"][:3]]
        lines.append(f"・{gem.status()}")
    except Exception as e:
        print("AIの組織のエラー:", e)
    try:
        hl = run_health()
        if hl:
            lines += ["━━━━━━━━━━", hl]
    except Exception:
        pass
    lines.append("※AIの予想であり利益を保証しません")
    word = plan[0] if plan else "様子見"
    title = (f"{'🟢' if word == '買い' else '🔴'} 今日のUSD/JPY: {word} 信頼度 {conf}%" if plan
             else f"⚪ 今日のUSD/JPY: 様子見 (下値 {sup:.2f} / 上値 {res:.2f} で待つ)")
    return title, "\n".join(lines)


def run(dry=False):
    from . import notify, scan
    h = rg.prepare(scan.load_recent())
    learn = load_learning()
    reviews = review(h, learn)
    learn = load_learning()
    a = analyze(h, learn)
    fc = forecast(h, a, learn) if not dry else {**{"price": h.Close.iat[-1], "lead": a["coord"]["lead"], "verdict": a["coord"]["verdict"],
                                                  "score": a["coord"]["score"], "exp_low": 0, "exp_high": 0,
                                                  "support": a["views"]["レンジ担当"]["support"], "resistance": a["views"]["レンジ担当"]["resistance"]}}
    title, body = morning_message(h, a, fc, reviews, learn)
    print(title); print(body)
    if not dry:
        notify.send(title, body, color=3447003)
