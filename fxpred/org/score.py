"""検証部: 専門AIを23年で採点し、会議での重みを決める。
python -m fxpred.org.score  → config/org_weights.json と reports/org.md を書く

採点: 日本時間10時の意見と、そこから1日先・5日先のドル円の動きの順位相関 (IC)。
重みの決め方 (たまたまを拾わないように):
  1) 作る期間 (2004-2015) で、理屈どおりの向きに効いている (1日先と5日先の IC の平均 > 0.01)
  2) 確かめる期間 (2016-) でも同じ向き (平均 > 0)
  両方を満たす専門AIだけ、作る期間の IC に比例した重み (最大1) を持つ。それ以外は 0 (意見は記録・説明に使う)
※ 2) で確かめる期間を見ているので、合格した専門AIの成績も「確かめる期間を一度見た」ものになる。影の運用で最終確認する"""
import json
import os

import numpy as np
import pandas as pd

from . import analysts as an, data as dt

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
W_PATH = os.path.join(ROOT, "config", "org_weights.json")


def forward(uj: pd.Series, n: int) -> pd.Series:
    return (np.log(uj).shift(-n) - np.log(uj)) * 1e4      # ベーシスポイント (0.01%)


def ic(a: pd.Series, r: pd.Series, mask) -> tuple[float, int]:
    k = mask & a.notna() & r.notna() & (a != 0)
    if k.sum() < 100:
        return np.nan, int(k.sum())
    return float(np.corrcoef(a[k].rank(), r[k].rank())[0, 1]), int(k.sum())


def build(h: pd.DataFrame):
    uj = dt.usdjpy_jst10(h)
    x = dt.at_jst10(dt.fetch(max_age_h=0), uj.index)
    b = an.features(x, uj, h)
    keep = b.index >= "2004-06-01"
    return b[keep], uj[keep]


def meeting_series(b: pd.DataFrame, w: dict) -> pd.DataFrame:
    """日ごとの会議の結論: 部門ごとの重み付き平均 → 全体"""
    out = pd.DataFrame(index=b.index)
    depts = sorted({c.split("|")[0] for c in b})
    tot = 0
    for d in depts:
        cols = [c for c in b if c.startswith(d + "|") and w.get(c, 0) > 0]
        if not cols:
            continue
        ww = np.array([w[c] for c in cols])
        out[d] = (b[cols].fillna(0) * ww).sum(axis=1) / ww.sum()
        tot += ww.sum()
        out[d + "_w"] = ww.sum()
    dcols = [d for d in depts if d in out]
    out["全体"] = sum(out[d] * out[d + "_w"] for d in dcols) / tot if dcols else 0.0
    return out


def trade_test(sig: pd.Series, uj: pd.Series, th: float, n: int, cost_bp=0.5):
    """意見が th 以上の日に、その向きに n 日持つ (重ならないように n 日ごと)。1回あたりの平均 bp・勝率"""
    r = forward(uj, n)
    s = sig.where(sig.abs() >= th, 0)
    res, i, idx = [], 0, s.index
    while i < len(idx):
        if s.iat[i] != 0 and not np.isnan(r.iat[i]):
            res.append((idx[i], np.sign(s.iat[i]) * r.iat[i] - cost_bp))
            i += n
        else:
            i += 1
    t = pd.Series(dict(res))
    return t


def run():
    h = pd.read_pickle(os.path.join(ROOT, "reports", "_h.pkl"))
    b, uj = build(h)
    r1, r5 = forward(uj, 1), forward(uj, 5)
    yrs = b.index.year
    B, V = yrs <= 2015, yrs >= 2016
    rows, w = [], {}
    for c in b:
        i1b, nb = ic(b[c], r1, B); i5b, _ = ic(b[c], r5, B)
        i1v, nv = ic(b[c], r1, V); i5v, _ = ic(b[c], r5, V)
        i5n, _ = ic(b[c], r5, yrs >= 2022)
        ib, iv = np.nanmean([i1b, i5b]), np.nanmean([i1v, i5v])
        ok = (ib > 0.01) and (iv > 0)
        w[c] = round(float(min(ib / 0.04, 1.0)), 2) if ok else 0.0
        rows.append(dict(部門=c.split("|")[0], 専門AI=c.split("|")[1], 作る1日=i1b, 作る5日=i5b, 確かめ1日=i1v, 確かめ5日=i5v,
                         直近22年5日=i5n, 日数=nb + nv, 合格="○" if ok else "", 重み=w[c]))
    R = pd.DataFrame(rows)
    # 会議の方法を比べる: 全員同じ重み (多数決) / 合格者だけ / 一番良い1人
    eq = {c: 1.0 for c in b}
    m_eq, m_w = meeting_series(b, eq), meeting_series(b, w)
    best = R.sort_values("作る5日", ascending=False).iloc[0]
    best_c = f"{best.部門}|{best.専門AI}"
    comp = []
    for name, sig in (("全員の多数決 (同じ重み)", m_eq["全体"]), ("合格した専門AIだけ (重み付き)", m_w["全体"]),
                      (f"作る期間で一番良い1人 ({best.専門AI})", b[best_c])):
        for n in (1, 5):
            r = forward(uj, n)
            i_b, _ = ic(sig, r, B); i_v, _ = ic(sig, r, V)
            t = trade_test(sig, uj, 0.25, n)
            tb, tv = t[t.index.year <= 2015], t[t.index.year >= 2016]
            comp.append(dict(会議の方法=name, 先=f"{n}日", 作るIC=i_b, 確かめIC=i_v,
                             作る_回数=len(tb), 作る_平均bp=tb.mean(), 作る_勝率=(tb > 0).mean(),
                             確かめ_回数=len(tv), 確かめ_平均bp=tv.mean(), 確かめ_勝率=(tv > 0).mean()))
    C = pd.DataFrame(comp)
    os.makedirs(os.path.dirname(W_PATH), exist_ok=True)
    json.dump({"weights": w, "note": "fxpred/org/score.py で作成。0 = 判断に入れない (記録と説明のみ)",
               "updated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")}, open(W_PATH, "w"), ensure_ascii=False, indent=1)
    write_report(R, C)
    return R, C


def write_report(R, C):
    f = lambda v: "" if pd.isna(v) else f"{v:+.3f}"
    L = ["# AI の組織: 専門AIの採点 (23年)", "",
         "毎日 日本時間10時の意見と、その後1日・5日のドル円の動きの順位相関 (IC)。0 = 関係なし、+ = 意見どおりに動いた。",
         "為替の先行きでは IC 0.02〜0.05 でも意味のある大きさ。作る期間 2004-2015 / 確かめる期間 2016-。",
         "向きは「よく言われる理屈」で固定し、データに合わせて入れ替えていない。", "",
         "| 部門 | 専門AI | 作る 1日 | 作る 5日 | 確かめ 1日 | 確かめ 5日 | 2022- 5日 | 合格 | 重み |", "|---|---|---|---|---|---|---|---|---|"]
    for r in R.itertuples():
        L.append(f"| {r.部門} | {r.専門AI} | {f(r.作る1日)} | {f(r.作る5日)} | {f(r.確かめ1日)} | {f(r.確かめ5日)} | {f(r.直近22年5日)} | {r.合格} | {r.重み} |")
    L += ["", "## 会議の方法の比べ方", "",
          "意見の強さが0.25以上の日に、その向きに1日 (または5日) 持った場合。往復0.5bp (約0.8pips) を引く。重ならないように数える。", "",
          "| 会議の方法 | 先 | 作るIC | 確かめIC | 作る 回数 | 作る 平均bp | 作る 勝率 | 確かめ 回数 | 確かめ 平均bp | 確かめ 勝率 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in C.itertuples():
        L.append(f"| {r.会議の方法} | {r.先} | {f(r.作るIC)} | {f(r.確かめIC)} | {r.作る_回数} | {r.作る_平均bp:+.1f} | {r.作る_勝率:.0%} | "
                 f"{r.確かめ_回数} | {r.確かめ_平均bp:+.1f} | {r.確かめ_勝率:.0%} |")
    open(os.path.join(ROOT, "reports", "org.md"), "w").write("\n".join(L) + "\n")


# ---------------------------------------------------------------- 相場分析部 (1時間足・毎時) の採点
def run_hourly():
    """相場分析部の専門AI (chart.py) を毎時の意見で採点。4時間先・24時間先 (次の足の始値から)"""
    from . import chart
    h = pd.read_pickle(os.path.join(ROOT, "reports", "_h.pkl"))
    b = chart.features(h)
    lo = np.log(h.Open)
    f4, f24 = (lo.shift(-1 - 4) - lo.shift(-1)) * 1e4, (lo.shift(-1 - 24) - lo.shift(-1)) * 1e4
    yrs = h.index.year
    B, V = yrs <= 2015, yrs >= 2016
    rows, w = [], {}
    for c in b:
        a = b[c]
        i4b, nb = ic(a, f4, B); i24b, _ = ic(a, f24, B)
        i4v, nv = ic(a, f4, V); i24v, _ = ic(a, f24, V)
        i24n, _ = ic(a, f24, yrs >= 2022)
        ib, iv = np.nanmean([i4b, i24b]), np.nanmean([i4v, i24v])
        # 合図の向きに24時間持った場合の平均 (pips 換算。重ならないよう24本ごと)
        sig = np.sign(a.values)
        pips = []
        i = 0
        r24 = (h.Open.shift(-25) - h.Open.shift(-1)).values * 100
        while i < len(sig):
            if sig[i] != 0 and not np.isnan(r24[i]):
                pips.append((yrs[i], sig[i] * r24[i] - 0.8)); i += 24
            else:
                i += 1
        p = pd.DataFrame(pips, columns=["y", "p"])
        # 毎時の専門AIは、IC に加えて「合図の向きに持った損益」が両方の期間でプラスであること
        ok = (ib > 0.01) and (iv > 0) and p.p[p.y <= 2015].mean() > 0 and p.p[p.y >= 2016].mean() > 0
        w[c] = round(float(min(ib / 0.04, 1.0)), 2) if ok else 0.0
        rows.append(dict(部門=c.split("|")[0], 専門AI=c.split("|")[1], 作る4h=i4b, 作る24h=i24b, 確かめ4h=i4v, 確かめ24h=i24v,
                         直近22年24h=i24n, 年間回数=len(p) / 22.9, 作る_pips=p.p[p.y <= 2015].mean(), 確かめ_pips=p.p[p.y >= 2016].mean(),
                         合格="○" if ok else "", 重み=w[c]))
    R = pd.DataFrame(rows)
    allw = json.load(open(W_PATH)) if os.path.exists(W_PATH) else {"weights": {}}
    allw.setdefault("weights", {}).update(w)
    json.dump(allw, open(W_PATH, "w"), ensure_ascii=False, indent=1)
    f = lambda v: "" if pd.isna(v) else f"{v:+.3f}"
    L = ["", "## 相場分析部の専門AI (1時間足で毎時)", "",
         "毎時の意見と、その後4時間・24時間のドル円の動きの順位相関 (IC)。合図の向きに24時間持った場合の平均 pips (往復0.8pips を引く、重ならないよう数える) も載せる。合格は IC と pips の両方が両期間でプラス。", "",
         "| 部門 | 専門AI | 作る 4h | 作る 24h | 確かめ 4h | 確かめ 24h | 2022- 24h | 年間回数 | 作る pips | 確かめ pips | 合格 | 重み |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in R.itertuples():
        L.append(f"| {r.部門} | {r.専門AI} | {f(r.作る4h)} | {f(r.作る24h)} | {f(r.確かめ4h)} | {f(r.確かめ24h)} | {f(r.直近22年24h)} | "
                 f"{r.年間回数:.0f} | {r.作る_pips:+.1f} | {r.確かめ_pips:+.1f} | {r.合格} | {r.重み} |")
    path = os.path.join(ROOT, "reports", "org.md")
    txt = open(path).read().split("\n## 相場分析部の専門AI")[0].rstrip("\n") + "\n"
    open(path, "w").write(txt + "\n".join(L) + "\n")
    return R


if __name__ == "__main__":
    import sys
    if "--hourly" in sys.argv:
        pd.set_option("display.width", 250)
        print(run_hourly().round(3).to_string(index=False))
        raise SystemExit
    R, C = run()
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
    print(C.round(3).to_string(index=False))
