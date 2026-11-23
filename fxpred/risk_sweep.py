"""1回のリスク (資金に対する損切り時の損失の割合) を変えて、資産の増え方が最大になる点を探す。
対象: 損小利大型 E2 (急変に付いていく 4倍・損切1ATR/利確4ATR・流れ方向 + 指標直後の順張り) と 安定型 E。
- 実際の順番での23年の結果
- 並べ替え (順番をランダムに) と ブロック並べ替え (連敗がかたまる時期を保つため、20回ずつのかたまりで並べ替え) を各5000回
- 国内の個人向け規制 (レバレッジ25倍) あり / なし"""
import os

import numpy as np
import pandas as pd

from . import lab

OUT = os.path.join(os.path.dirname(__file__), "..", "reports", "risk_sweep.md")
RISKS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
YEARS = 22.9


def trades(h, a, kind="E2"):
    d = lab.prep(h)
    win = (a.jst_hour >= 10) | (a.jst_hour < 1)
    if kind == "E2":
        sf = lab.shock_follow(d, z_th=4, skip_mf=False, stop_atr=1.0, tgt_atr=4.0)
        ev = a[(a.key == "H5") & win]
    else:
        sf = lab.shock_follow(d, z_th=3.5)
        ev = a[(a.key == "H5") & win & ~a.weekday.isin([0, 4])]
    t = pd.concat([sf[["time", "pips", "risk"]], ev[["time", "pips"]].assign(risk=ev.risk_pips.values)]).sort_values("time")
    t["R"] = (t.pips - 0.3) / t.risk.clip(lower=3)              # 研究の0.5pipsに手動発注のずれ0.3pipsを追加
    t["price"] = d.Close.reindex(t.time).values
    t["cap"] = 25 * t.risk.clip(lower=3) / 100 / t.price         # レバレッジ25倍で取れる最大のリスク
    return t.reset_index(drop=True)


def path(R, cap, f, use_cap=True):
    eff = np.minimum(f, cap) if use_cap else np.full_like(R, f)
    eq = np.cumprod(np.maximum(1 + eff * R, 1e-9))
    return eq, (1 - eq / np.maximum.accumulate(eq)).max()


def sweep(t, n_sim=4000, block=10, seed=0, use_cap=True):
    """最終資金のばらつき: 取引を重複ありで選び直す (ブートストラップ。成績そのものの運・不運を含む)。
    連敗がかたまる時期を残すため、10回ずつのかたまりで選び直す。最大下落は同じ並びから測る。
    1年・3年・5年後の資金も、その期間の取引回数ぶんを同じ方法で選び直して出す"""
    rng = np.random.default_rng(seed)
    R, cap = t.R.values, t.cap.values
    n = len(R)
    per_year = n / YEARS

    def boot(k):
        starts = rng.integers(0, n - block, int(np.ceil(k / block)))
        return np.concatenate([np.arange(s_, s_ + block) for s_ in starts])[:k]

    rows = []
    for f in RISKS:
        eq, dd = path(R, cap, f, use_cap)
        res = {}
        for yrs in (1, 3, 5, YEARS):
            k = max(int(round(per_year * yrs)), 1)
            fin, dds = [], []
            for _ in range(n_sim):
                idx = boot(k)
                e, d_ = path(R[idx], cap[idx], f, use_cap)
                fin.append(e[-1]); dds.append(d_)
            res[yrs] = (np.array(fin), np.array(dds))
        f23, d23 = res[YEARS]
        f1, d1 = res[1]
        f3, _ = res[3]
        f5, _ = res[5]
        lev = np.minimum(f, cap) * t.price.values / (t.risk.clip(lower=3).values / 100)
        rows.append({"1回のリスク": f"{f:.1%}", "_f": f,
                     "実際の23年": f"{eq[-1] * 100:,.0f}万 (年率{eq[-1] ** (1 / YEARS) - 1:+.1%}, 最大下落{dd:.0%})",
                     "23年後 中央値(万)": round(np.median(f23) * 100), "23年後 悪い方10%(万)": round(np.quantile(f23, .1) * 100),
                     "1年後 中央値": f"{np.median(f1):.2f}倍", "1年後 悪い方10%": f"{np.quantile(f1, .1):.2f}倍",
                     "3年後 中央値": f"{np.median(f3):.2f}倍", "3年で2倍以上の確率": f"{(f3 >= 2).mean():.0%}",
                     "5年で2倍以上の確率": f"{(f5 >= 2).mean():.0%}", "5年後に元本割れの確率": f"{(f5 < 1).mean():.0%}",
                     "1年の最大下落 中央値": f"{np.median(d1):.0%}", "23年の最大下落 中央値": f"{np.median(d23):.0%}",
                     "23年で一度は半分以下になる確率": f"{(d23 >= .5).mean():.0%}",
                     "_med": np.median(f23), "_q10": np.quantile(f23, .1),
                     "平均レバレッジ": f"{lev.mean():.1f}倍" if use_cap else "-"})
    return pd.DataFrame(rows)


def kelly(t):
    R = t.R.values
    fs = np.linspace(0.001, 0.4, 400)
    g = [np.mean(np.log(np.maximum(1 + f * R, 1e-9))) for f in fs]
    return fs[int(np.argmax(g))], R.mean(), (R > 0).mean(), R[R > 0].mean(), R[R <= 0].mean()


def run(h=None, a=None):
    if h is None:
        h = pd.read_pickle(os.path.join(os.path.dirname(OUT), "_h.pkl"))
        a = pd.read_pickle(os.path.join(os.path.dirname(OUT), "_signals_all.pkl"))
    md = ["# リスクの割合と資産の増え方 (23年, 100万円から)", ""]
    for kind, label in (("E2", "損小利大型 E2"), ("E", "安定型 E")):
        t = trades(h, a, kind)
        k, m, wr, aw, al = kelly(t)
        md += [f"## {label}: {len(t)}回 (年{len(t) / YEARS:.0f}回) 勝率{wr:.0%} 勝ちの平均{aw:+.2f}R 負けの平均{al:+.2f}R 1回平均{m:+.3f}R", "",
               f"理論上、長期の増え方が最大になる1回のリスク (ケリー基準): **{k:.1%}** (この半分〜3分の1が一般的な目安)", ""]
        for use_cap, cap_label in ((True, "国内のレバレッジ規制 (25倍) あり"), (False, "規制なし (理論値)")):
            s = sweep(t, use_cap=use_cap)
            best = s.loc[s._med.idxmax()]
            safe = s.loc[s._q10.idxmax()]
            md += [f"### {cap_label}", "", s.drop(columns=["_f", "_med", "_q10"]).to_markdown(index=False), "",
                   f"- 23年後の中央値が最大: 1回のリスク **{best['1回のリスク']}** (中央値 {best['_med'] * 100:,.0f}万円)",
                   f"- 運が悪い場合 (悪い方10%) でも最も増える: 1回のリスク **{safe['1回のリスク']}** (悪い方10%で {safe['_q10'] * 100:,.0f}万円)", ""]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(md) + "\n")
    return "\n".join(md)


if __name__ == "__main__":
    print(run())
