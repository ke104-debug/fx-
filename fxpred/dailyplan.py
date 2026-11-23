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
    day_rng = float((h.High.rolling(24).max() - h.Low.rolling(24).min()).iloc[-480:].median())
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
        row = {"time_utc": r.time_utc, "date": r.date, "verdict": r.verdict, "actual": actual, "move": round(move, 2),
               "hit_dir": hit_dir, "high": round(hi, 2), "low": round(lo, 2), "range_ratio": round(rng_ratio, 2),
               "in_range": in_range, "support_broken": sup_broken, "resistance_broken": res_broken,
               **{f"hit_{k}": v for k, v in spec.items()}, "lessons": " / ".join(lessons) or "予想どおり"}
        pd.DataFrame([row]).to_csv(REVIEW, mode="a", header=not os.path.exists(REVIEW), index=False)
        out.append(row)
    if out:
        improve(learn)
    return out


def improve(learn: dict) -> dict:
    """直近の振り返りから重みと値幅補正を更新する (少ない件数では大きく変えない)"""
    rv = pd.read_csv(REVIEW).tail(30)
    notes = []
    for name in ("急変担当", "レンジ担当", "トレンド担当"):
        col = rv.get(f"hit_{name}")
        if col is None:
            continue
        calls = col.dropna().astype(str).map({"True": 1, "False": 0}).dropna()
        if len(calls) >= 5:
            acc = calls.mean()
            wgt = float(np.clip(0.5 + (acc - 0.5) * 2, 0.5, 1.5))   # 正解率50%で1.0倍、75%で1.5倍、25%で0.5倍
            old = learn["weights"].get(name, 1.0)
            learn["weights"][name] = round(0.7 * old + 0.3 * wgt, 2)
            notes.append(f"{name}: 直近{len(calls)}回の正解率 {acc:.0%} → 重み {learn['weights'][name]}")
    if len(rv) >= 5:
        target = float(np.clip(rv.range_ratio.median(), 0.7, 1.5))
        learn["range_scale"] = round(0.7 * learn.get("range_scale", 1.0) + 0.3 * learn.get("range_scale", 1.0) * target, 3)
        notes.append(f"予想値幅の補正: x{learn['range_scale']}")
    learn["notes"] = notes
    learn["updated"] = f"{pd.Timestamp.now(tz=JST):%Y-%m-%d %H:%M}"
    json.dump(learn, open(LEARN, "w"), ensure_ascii=False, indent=1)
    return learn


def morning_message(h: pd.DataFrame, a: dict, fc: dict, reviews: list[dict], learn: dict) -> tuple[str, str]:
    lines = []
    if reviews:
        r = reviews[-1]
        lines += ["【昨日の振り返り】",
                  f"予想 {r['verdict']} → 実際 {r['actual']} ({r['move']:+.2f}円)  {'○' if r['hit_dir'] else '×'}",
                  f"値幅 {r['low']}〜{r['high']} (予想の{r['range_ratio']}倍) {'○' if r['in_range'] else '×'}",
                  f"反省: {r['lessons']}"]
        if learn.get("notes"):
            lines.append("改善: " + " / ".join(learn["notes"]))
        allr = pd.read_csv(REVIEW)
        lines.append(f"通算: 方向 {allr.hit_dir.mean():.0%} ({len(allr)}日) / 値幅の的中 {allr.in_range.mean():.0%}")
        lines.append("")
    v = a["views"]
    ev = [e for e in rg.st.events(h.index[-50:].append(pd.DatetimeIndex([h.index[-1] + timedelta(days=2)])))
          if timedelta(0) < e - h.index[-1] <= timedelta(hours=24)]
    lines += ["【今日の予想】",
              f"現在 {fc['price']:.2f}円 / 相場状態: {a['regime']} (主担当: {fc['lead']})",
              f"方向: {fc['verdict']} (まとめ役のスコア {fc['score']:+})",
              f"予想値幅: {fc['exp_low']:.2f}〜{fc['exp_high']:.2f}",
              f"下値 {fc['support']:.2f} / 上値 {fc['resistance']:.2f}",
              "重要指標: " + ("、".join(f"{e.tz_convert(JST):%H:%M}" for e in ev) if ev else "主要な発表なし (FOMC・雇用統計基準)"),
              "", "【担当AIの見立て】"]
    for name in ("トレンド担当", "レンジ担当", "急変担当"):
        lines.append(f"■ {name}{' (主担当)' if name == fc['lead'] else ''}")
        lines += ["・" + x for x in v[name]["lines"]]
    lines += ["", "※AIの予想であり利益を保証しません。チャンスと判断した時は別途通知します"]
    return f"USD/JPY 今日の予想 {pd.Timestamp.now(tz=JST):%m/%d}", "\n".join(lines)


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
