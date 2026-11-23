"""1回あたりのリスク (資金に対する損失上限) を変えたときの増え方と危険度を調べる。
過去の取引結果 (R倍数 = 損益 ÷ 損切り幅) を使い、資金に複利で当てはめる。レバレッジ25倍の上限も反映。"""
import numpy as np
import pandas as pd

from . import portfolio as pf
from . import strategies as st

RISKS = (0.01, 0.02, 0.03, 0.05, 0.10)


def trades_for(names):
    h = st.indicators(st.load()); h["trend_up"] = h.Close.ewm(span=480).mean().diff(24) > 0
    ev = st.events(h.index); pos = ev.searchsorted(h.index, side="right") - 1
    prev = pd.Series([ev[p] if p >= 0 else pd.NaT for p in pos], index=h.index)
    h["h_since_event"] = (h.index.to_series() - prev).dt.total_seconds() / 3600 - 1
    diff = pf.rate_diff(h.index)
    out = []
    for n in names:
        _, tr = pf.run_hourly(h, pf.HOURLY[n], diff=diff)
        out.append(tr.assign(作戦=n))
    t = pd.concat(out).sort_values("exit").reset_index(drop=True)
    return t, (h.index[-1] - h.index[0]).days / 365.25


def grow(t, f, eq=1.0):
    path, peak, mdd = [], eq, 0
    for R, dist, px in zip(t.R, t.dist, t.price):
        f_eff = min(f, pf.MAX_LEV * dist / px)        # 25倍を超える取引量にはしない
        eq *= max(1 + f_eff * R, 0)
        peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak); path.append(eq)
    return eq, mdd, path


def monte_carlo(t, f, n_trades, sims=5000, seed=0):
    rng = np.random.default_rng(seed)
    R, cap = t.R.values, (pf.MAX_LEV * t.dist / t.price).values
    fin, mdd = np.empty(sims), np.empty(sims)
    for s in range(sims):
        k = rng.integers(0, len(R), n_trades)
        g = np.maximum(1 + np.minimum(f, cap[k]) * R[k], 0)
        eqs = np.cumprod(g)
        fin[s] = eqs[-1]; mdd[s] = (1 - eqs / np.maximum.accumulate(np.r_[1, eqs])[1:]).max()
    return fin, mdd


def run():
    sets = {"H3 のみ": ["H3 東京レンジのロンドン抜け"],
            "H3+H4+H5 (3作戦を同時に)": ["H3 東京レンジのロンドン抜け", "H4 レンジ逆張り+流れフィルタ", "H5 指標発表直後の順張り"]}
    for label, names in sets.items():
        t, years = trades_for(names)
        per_half = int(round(len(t) / years / 2))
        print(f"\n■ {label}: {len(t)}回 / {years:.1f}年 (半年あたり約{per_half}回) / 1回平均 {t.R.mean():+.3f}R")
        rows = []
        for f in RISKS:
            fin, mdd, _ = grow(t, f)
            mc_f, mc_d = monte_carlo(t, f, per_half)
            rows.append({"1回のリスク": f"{f:.0%}", "2.8年の結果": f"{fin - 1:+.0%}", "最大下落": f"{mdd:.0%}",
                         "半年: 中央値": f"{np.median(mc_f) - 1:+.0%}", "半年: 上位10%": f"{np.quantile(mc_f, .9) - 1:+.0%}",
                         "半年で+150%以上": f"{(mc_f >= 2.5).mean():.1%}", "半年で資金半減以下": f"{(mc_f <= 0.5).mean():.1%}",
                         "半年で40%以上の下落を経験": f"{(mc_d >= 0.4).mean():.1%}"})
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    run()
