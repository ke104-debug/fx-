"""精度を上げるための研究用の計算 (20年データで、運用とは別に GitHub 上でじっくり実行する)

  1. パラメータ感度   : 主な作戦のパラメータを少しずつ動かして全組み合わせを回し、周辺も安定している設定を探す
  2. ウォークフォワード: 過去5年で設定を選び、次の1年で確かめる。1年ずつずらして繰り返す
  3. 年代別の成績     : リーマン、アベノミクス、コロナ、2022年以降の円安と介入 などで分ける
  4. モンテカルロ     : 取引の順番を並べ替えて、最大連敗と最悪の資金の減り方の分布を出す (ロット・当日停止ルールの根拠)
  5. コスト悪化       : スプレッドを広げ、約定を不利にずらしても期待値が残るか
  6. 足の中の順番     : 同じ足で損切りと利確の両方に触れた取引の数と、損切り優先 (保守的) の影響
  7. フィルターの切り分け: ノートレード判定・相場状態での切り替え・流れの方向・信頼度の基準などを1つずつオン/オフ

すべての作戦の候補シグナルを「フィルターをかける前」の状態で1回だけ作り、その表を使い回して速く計算する。
"""
import itertools
import os

import numpy as np
import pandas as pd

from . import diagnose, edge, portfolio as pf, regime as rg, scan, shock as sh, strategies as st

OUT = os.path.join(os.path.dirname(__file__), "..", "reports", "research.md")
ERAS = [("2004-07 リーマン前", 2004, 2007), ("2008-09 リーマン", 2008, 2009), ("2010-12 円高・2011介入", 2010, 2012),
        ("2013-15 アベノミクス", 2013, 2015), ("2016-19 低変動", 2016, 2019), ("2020-21 コロナ", 2020, 2021),
        ("2022-26 円安・介入", 2022, 2026)]
HOLD = {"H3": 14, "H4": 48, "H5": 24, "S": 72}
COST = 0.5  # 往復 pips


def outcome(h, j, side, stop, tgt, hold, cost=COST, slip=0.0, target_first=False):
    """次の足の始値で入り、損切り/利確/期限で決済。slip: 入りと損切り時に不利にずらす pips。
    同じ足で両方に触れたら損切り優先 (target_first=True で利確優先にした場合の上限も出せる)"""
    O, H, L, C = h.Open.values, h.High.values, h.Low.values, h.Close.values
    e = O[j] + side * slip / 100
    amb = False
    end = min(j + hold, len(h)) - 1
    for k in range(j, end + 1):
        hs = (side == 1 and L[k] <= stop) or (side == -1 and H[k] >= stop)
        ht = (side == 1 and H[k] >= tgt) or (side == -1 and L[k] <= tgt)
        if hs and ht:
            amb = True
        if hs and not (ht and target_first):
            return side * (stop - e) * 100 - cost - slip, "損切り", amb
        if ht:
            return side * (tgt - e) * 100 - cost, "利確", amb
    return side * (C[end] - e) * 100 - cost, "期限切れ", amb


def candidates(h: pd.DataFrame) -> pd.DataFrame:
    """全作戦 x 全相場状態の候補シグナル (フィルター前)。各フィルターの判定材料を列として持つ"""
    raw = h[["Open", "High", "Low", "Close"]]
    up = h.trend_up.values
    events = st.events(h.index)
    rows = []

    def add(i, key, side, stop, tgt, aligned=True):
        if i + 1 >= len(h):
            return
        t = h.index[i]
        sub = h.iloc[max(0, i - 1500):i + 1]
        tech = diagnose.technical(raw.iloc[max(0, i - 1500):i + 1])
        tf = {k: tech[k]["trend"] for k in ("15分足", "1時間足", "4時間足")}
        reg = rg.NAMES[h.regime.iat[i]]
        anom = edge.anomaly(sub)
        n_ok, ok, ng = edge.confluence(side, tf, scan.near_level(sub, side), 0.5, None, reg, True)
        ok = [x for x in ok if not x.startswith("AIモデル")]      # 20年分の AIモデルの確率は再現しない
        warn = edge.no_trade(sub, t + pd.Timedelta(hours=1), reg, tf, anom)
        serious = any(w.startswith(("重要指標", "急変の最中", "週末")) for w in warn)
        if tgt is None:
            tgt = h.Close.iat[i] + side * 2 * abs(h.Close.iat[i] - stop)
        pips, res, amb = outcome(h, i + 1, side, stop, tgt, HOLD[key])
        risk_pips = max(abs(h.Open.iat[i + 1] - stop) * 100, 1.0)
        pips_tf, _, _ = outcome(h, i + 1, side, stop, tgt, HOLD[key], target_first=True)
        rows.append(dict(time=t, key=key, regime=reg, side=side, aligned=aligned, score_ok="|".join(x.split(" ")[0] for x in ok),
                         n_warn=len(warn), serious=serious, awake=scan.awake(t + pd.Timedelta(hours=1), events),
                         stop=stop, target=tgt, i=i, pips=pips, risk_pips=risk_pips, result=res, ambiguous=amb, pips_target_first=pips_tf))

    for i in range(700, len(h) - 2):
        r = pf.h_tokyo_break(h, i)
        if r: add(i, "H3", r[0], r[1], r[2])
        r = st.range_reversal(h, i, both_sides=True)                       # 流れフィルタをかける前
        if r: add(i, "H4", r[0], r[1], r[2], aligned=(r[0] == 1) == bool(up[i]))
        r = pf.h_event_momo(h, i)
        if r: add(i, "H5", r[0], r[1], r[2])
    for side in (1, -1):
        t = sh.find(h, side=side)
        for r in t.itertuples():
            i = h.index.get_loc(r.入り) - 1
            add(i, "S", side, r.損切り, r.利確, aligned=bool(up[h.index.get_loc(r.検知)]) == (side == 1))
    a = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    a["era"] = pd.cut(a.time.dt.year, [e[1] - 1 for e in ERAS] + [ERAS[-1][2]], labels=[e[0] for e in ERAS])
    return a


def score(a: pd.DataFrame, weights=None, penalties=True) -> pd.Series:
    w = weights or edge.COND_WEIGHTS
    s = pd.Series(edge.BASE, index=a.index, dtype=float)
    for c, v in w.items():
        s += np.where(a.score_ok.str.contains(c), v, 0)
    if penalties:
        s -= 10 * a.n_warn
    return s.clip(0, 100)


def live_mask(a: pd.DataFrame, off: set = frozenset()) -> pd.Series:
    """今の運用ルールで通知されるシグナル。off に入れたフィルターは外す"""
    m = pd.Series(True, index=a.index)
    off = set(off)
    if "相場状態で作戦を切り替え" not in off:
        m &= pd.Series([(k, r) in scan.RULES for k, r in zip(a.key, a.regime)], index=a.index)
    if "流れの方向だけ (レンジ逆張り・急変の反動)" not in off:
        m &= a.aligned | ~a.key.isin(["H4", "S"])
    if "ノートレード判定" not in off:
        m &= ~a.serious
    if "信頼度の基準 (65%)" not in off:
        sc = score(a, penalties="警戒事項の減点" not in off)
        m &= (sc >= edge.SEND_AT) | (a.key == "S")
    if "通知時間 (10時〜翌1時)" not in off:
        m &= a.awake | (a.key == "S")
    return m


def stats(x: pd.DataFrame) -> dict:
    if x.empty:
        return {"回数": 0}
    eq = x.pips.cumsum()
    return {"回数": len(x), "勝率": f"{(x.pips > 0).mean():.0%}", "平均pips": round(x.pips.mean(), 1),
            "合計pips": round(x.pips.sum()), "最大DD(pips)": round((eq.cummax() - eq).max())}


# ---------------------------------------------------------------- 7. フィルターの切り分け
FILTERS = ["相場状態で作戦を切り替え", "流れの方向だけ (レンジ逆張り・急変の反動)", "ノートレード判定",
           "信頼度の基準 (65%)", "警戒事項の減点", "通知時間 (10時〜翌1時)"]


def ablation(a: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = live_mask(a)
    for f in [None] + FILTERS:
        m = live_mask(a, {f} if f else set())
        x = a[m]
        r = {"設定": "今の運用 (全部オン)" if f is None else f"「{f}」をオフ", **stats(x)}
        eras = x.groupby("era", observed=True).pips.mean()
        b_eras = a[base].groupby("era", observed=True).pips.mean()
        r["年代で改善した数"] = "" if f is None else f"{(eras.reindex(b_eras.index) > b_eras).sum()}/{len(b_eras)}"
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. 年代別
def by_era(a: pd.DataFrame) -> pd.DataFrame:
    x = a[live_mask(a)]
    rows = []
    for era, g in x.groupby("era", observed=True):
        rows.append({"年代": era, **stats(g), "作戦別の平均pips": {k: round(v, 1) for k, v in g.groupby("key").pips.mean().items()}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. モンテカルロ
def monte_carlo(a: pd.DataFrame, n=5000, seed=0) -> pd.DataFrame:
    x = a[live_mask(a)]
    R = (x.pips / x.risk_pips).values
    rng = np.random.default_rng(seed)
    streaks, dds = [], {r: [] for r in (0.005, 0.01, 0.02, 0.03)}
    for _ in range(n):
        p = rng.permutation(R)
        s = mx = 0
        for v in p:
            s = s + 1 if v <= 0 else 0; mx = max(mx, s)
        streaks.append(mx)
        for r in dds:
            eq = np.cumprod(1 + r * p)
            dds[r].append((1 - eq / np.maximum.accumulate(eq)).max())
    rows = [{"項目": "最大連敗", "中央値": int(np.median(streaks)), "悪い方5%": int(np.quantile(streaks, .95)), "最悪": int(max(streaks))}]
    for r, d in dds.items():
        rows.append({"項目": f"最大下落 (1回のリスク{r:.1%})", "中央値": f"{np.median(d):.0%}", "悪い方5%": f"{np.quantile(d, .95):.0%}",
                     "最悪": f"{max(d):.0%}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 5. コスト悪化
def cost_stress(h, a: pd.DataFrame) -> pd.DataFrame:
    x = a[live_mask(a)]
    rows = []
    for spread, slip in ((0.5, 0), (1.0, 0), (2.0, 0), (1.0, 1.0), (1.0, 2.0), (2.0, 3.0)):
        p = [outcome(h, r.i + 1, r.side, r.stop, r.target, HOLD[r.key], cost=spread, slip=slip)[0] for r in x.itertuples()]
        p = pd.Series(p)
        rows.append({"スプレッド(往復)": spread, "約定のずれ(入り・損切り)": slip, "平均pips": round(p.mean(), 1),
                     "合計pips": round(p.sum()), "勝率": f"{(p > 0).mean():.0%}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 6. 足の中の順番
def intrabar(a: pd.DataFrame) -> dict:
    x = a[live_mask(a)]
    amb = x[x.ambiguous]
    return {"判定が微妙な取引 (同じ足で両方に触れた)": f"{len(amb)}回 / {len(x)}回 ({len(amb) / max(len(x), 1):.1%})",
            "損切り優先 (採用している保守的な判定) の平均": round(x.pips.mean(), 2),
            "利確優先にした場合 (上限の目安) の平均": round(x.pips_target_first.mean(), 2)}


# ---------------------------------------------------------------- 1. 感度 と 2. ウォークフォワード
GRIDS = {
    "H4 レンジ逆張り": dict(n=[48, 72, 96, 120], target=[0.6, 0.7, 0.8], stop_atr=[0.5, 0.75, 1.0], adx=[20, 25, 30]),
    "H3 東京レンジのロンドン抜け": dict(wmin=[10, 15, 20], wmax=[45, 60, 80], rr=[1.5, 2.0, 2.5]),
    "S 急変の反動": dict(drop=[1.5, 2.0, 2.5], retrace=[0.3, 0.4, 0.5], buffer=[0.1, 0.2, 0.3]),
}


def run_param(h, name, p) -> pd.DataFrame:
    """1つのパラメータ組で作戦を全期間に流し、取引ごとの R (損益 ÷ 損切り幅) を返す"""
    rows = []
    if name.startswith("H4"):
        d = st.indicators(h[["Open", "High", "Low", "Close"]], n=p["n"])
        d["trend_up"] = h.trend_up.values
        for i in range(700, len(d) - 2):
            r = d.iloc[i]
            if not (4 * r.atr <= r.width <= 15 * r.atr) or r.adx >= p["adx"] or r.drift >= 0.5 * r.width:
                continue
            near = 0.15 * r.width
            for side in (1, -1):
                if side == 1 and not (r.Low <= r.sup + near and r.Close > r.sup and r.Close > r.Open): continue
                if side == -1 and not (r.High >= r.res - near and r.Close < r.res and r.Close < r.Open): continue
                if (side == 1) != bool(r.trend_up): continue
                stop = r.sup - p["stop_atr"] * r.atr if side == 1 else r.res + p["stop_atr"] * r.atr
                tgt = r.sup + p["target"] * r.width if side == 1 else r.res - p["target"] * r.width
                e = d.Open.iat[i + 1]
                if side * (e - stop) <= 0 or side * (tgt - e) <= 0: continue
                pips, _, _ = outcome(d, i + 1, side, stop, tgt, 48)
                rows.append((d.index[i], pips / max(abs(e - stop) * 100, 1)))
    elif name.startswith("H3"):
        for i in range(700, len(h) - 2):
            r = pf.session_break(h, i, (0, 6), (7, 11), p["wmin"], p["wmax"], p["rr"], 20)
            if r:
                e = h.Open.iat[i + 1]
                if r[0] * (e - r[1]) <= 0: continue
                pips, _, _ = outcome(h, i + 1, r[0], r[1], r[2], 14)
                rows.append((h.index[i], pips / max(abs(e - r[1]) * 100, 1)))
    else:
        t = sh.trend_aligned(h, drop=p["drop"], retrace=p["retrace"])
        for r in t.itertuples():
            rows.append((r.入り, r.R))
    return pd.DataFrame(rows, columns=["time", "R"])


def sensitivity_and_wf(h) -> tuple[dict, dict]:
    sens, wf = {}, {}
    for name, grid in GRIDS.items():
        keys = list(grid)
        combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
        res = {}
        for c in combos:
            res[tuple(c.values())] = run_param(h, name, c)
        # 1. 感度: 各組の平均R と、隣り合う組 (1つのパラメータだけ1段ずれ) を含めた平均 (台地の高さ)
        rows = []
        for c in combos:
            k = tuple(c.values())
            neigh = [k]
            for idx, key in enumerate(keys):
                vals = grid[key]; pos = vals.index(c[key])
                for dp in (-1, 1):
                    if 0 <= pos + dp < len(vals):
                        nk = list(k); nk[idx] = vals[pos + dp]; neigh.append(tuple(nk))
            own = res[k].R.mean() if len(res[k]) else np.nan
            plateau = np.nanmean([res[n].R.mean() for n in neigh if len(res[n])])
            rows.append({**c, "回数": len(res[k]), "平均R": round(own, 3), "周辺を含めた平均R": round(plateau, 3)})
        s = pd.DataFrame(rows).sort_values("周辺を含めた平均R", ascending=False)
        sens[name] = s
        # 2. ウォークフォワード: 過去5年で「周辺を含めた平均R」が最も高い組を選び、次の1年で確かめる
        years = sorted({t.year for df in res.values() for t in df.time})
        out = []
        for y in years:
            tr_lo, tr_hi = y - 5, y - 1
            if tr_lo < years[0]:
                continue
            def m(k, lo, hi):
                df = res[k]; x = df[(df.time.dt.year >= lo) & (df.time.dt.year <= hi)]
                return x.R.mean() if len(x) >= 10 else np.nan
            best, bv = None, -9
            for c in combos:
                k = tuple(c.values())
                neigh = [k] + [tuple(v if j != idx else grid[key][grid[key].index(c[key]) + dp] for j, v in enumerate(k))
                               for idx, key in enumerate(keys) for dp in (-1, 1)
                               if 0 <= grid[key].index(c[key]) + dp < len(grid[key])]
                v = np.nanmean([m(n, tr_lo, tr_hi) for n in neigh])
                if v > bv: best, bv = k, v
            test = res[best]; test = test[test.time.dt.year == y]
            cur = res[tuple(list(GRIDS_DEFAULT[name].values()))]; cur = cur[cur.time.dt.year == y]
            out.append({"検証の年": y, "選んだ設定": dict(zip(keys, best)), "その年の平均R": round(test.R.mean(), 3) if len(test) else None,
                        "今の設定の平均R": round(cur.R.mean(), 3) if len(cur) else None, "回数": len(test)})
        wf[name] = pd.DataFrame(out)
    return sens, wf


GRIDS_DEFAULT = {"H4 レンジ逆張り": dict(n=72, target=0.8, stop_atr=0.5, adx=25),
                 "H3 東京レンジのロンドン抜け": dict(wmin=15, wmax=60, rr=2.0),
                 "S 急変の反動": dict(drop=2.0, retrace=0.4, buffer=0.2)}


AUTO = ["ノートレード判定", "警戒事項の減点"]   # 自動で変えてよいフィルター (通知時間は生活の都合なので対象外)
FILTERS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "filters.json")


def decide(a: pd.DataFrame) -> tuple[dict, list[str]]:
    """フィルターをオフにした方が「全体の平均が良い」かつ「7つの年代のうち5つ以上で良い」かつ
    「最大の資金の減り方が悪化しない」時だけ、そのフィルターをオフにする。逆の時はオンに戻す"""
    import json
    cur = json.load(open(FILTERS_PATH)) if os.path.exists(FILTERS_PATH) else {}
    notes = []
    base = a[live_mask(a, {f for f in AUTO if cur.get(f) is False})]
    b_st, b_eras = stats(base), base.groupby("era", observed=True).pips.mean()
    for f in AUTO:
        off = {g for g in AUTO if cur.get(g) is False} | {f}
        on = off - {f}
        xo, xn = a[live_mask(a, off)], a[live_mask(a, on)]
        so, sn = stats(xo), stats(xn)
        eo, en = xo.groupby("era", observed=True).pips.mean(), xn.groupby("era", observed=True).pips.mean()
        better = (eo.reindex(en.index) > en).sum()
        n_eras = len(en)
        if so.get("平均pips", -99) > sn.get("平均pips", -99) and better >= max(3, int(np.ceil(n_eras * 5 / 7))) \
                and so.get("最大DD(pips)", 9e9) <= sn.get("最大DD(pips)", 0) * 1.05:
            if cur.get(f, True):
                notes.append(f"「{f}」をオフにします (平均 {sn['平均pips']:+} → {so['平均pips']:+} pips、{better}/{n_eras}年代で改善)")
            cur[f] = False
        elif not cur.get(f, True) and sn.get("平均pips", -99) >= so.get("平均pips", -99):
            notes.append(f"「{f}」をオンに戻します (オンの方が平均 {sn['平均pips']:+} pips で良い)")
            cur[f] = True
    return cur, notes


def run(h: pd.DataFrame | None = None, quick=False, apply=False) -> str:
    from . import longdata
    if h is None:
        d = pd.concat([longdata.load(), st.load()]).sort_index()
        d = d[~d.index.duplicated(keep="first")]
        h = rg.prepare(d)
    a = candidates(h)
    md = [f"# 精度向上の研究レポート", "", f"期間: {h.index[0]:%Y-%m-%d}〜{h.index[-1]:%Y-%m-%d} / 候補シグナル {len(a)}件 / コスト往復{COST}pips",
          "AIモデルの確率と VIX は20年分を再現できないため除外 (配点への影響は小さい)", ""]
    ab = ablation(a)
    md += ["## 7. フィルターの切り分け", "", ab.to_markdown(index=False), ""]
    cfg, notes = decide(a)
    md += ["### 自動反映の判断", ""] + ([f"- {n}" for n in notes] or ["- 変更なし (条件を満たすフィルターはなし)"]) + [""]
    if apply and notes:
        import json
        os.makedirs(os.path.dirname(FILTERS_PATH), exist_ok=True)
        json.dump(cfg, open(FILTERS_PATH, "w"), ensure_ascii=False, indent=1)
    md += ["## 3. 年代別の成績 (今の運用ルール)", "", by_era(a).to_markdown(index=False), ""]
    md += ["## 5. コスト悪化テスト", "", cost_stress(h, a).to_markdown(index=False), ""]
    md += ["## 6. 足の中の順番 (損切り優先の影響)", ""] + [f"- {k}: {v}" for k, v in intrabar(a).items()] + [""]
    md += ["## 4. モンテカルロ (取引の順番を5000回並べ替え)", "", monte_carlo(a).to_markdown(index=False), ""]
    if not quick:
        sens, wf = sensitivity_and_wf(h)
        for name in GRIDS:
            md += [f"## 1. パラメータ感度: {name} (周辺を含めた平均Rの上位10)", "", sens[name].head(10).to_markdown(index=False), "",
                   f"今の設定: {GRIDS_DEFAULT[name]}", "",
                   f"## 2. ウォークフォワード: {name} (過去5年で選び、次の1年で検証)", "", wf[name].to_markdown(index=False), ""]
            w = wf[name].dropna()
            if len(w):
                md.append(f"選んだ設定がその年に今の設定を上回った年: {(w['その年の平均R'] > w['今の設定の平均R']).sum()}/{len(w)}、"
                          f"選んだ設定の平均R {w['その年の平均R'].mean():+.3f} / 今の設定 {w['今の設定の平均R'].mean():+.3f}")
                md.append("")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(md) + "\n")
    a.to_csv(os.path.join(os.path.dirname(OUT), "research_signals.csv.gz"), index=False)
    summary = ab.to_string(index=False) + "\n\n自動反映: " + (" / ".join(notes) if notes else "変更なし")
    return summary


if __name__ == "__main__":
    import sys
    print(run(quick="--quick" in sys.argv, apply="--apply" in sys.argv))
