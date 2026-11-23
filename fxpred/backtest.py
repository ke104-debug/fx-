"""ウォークフォワード検証: 過去だけで学習 -> 未来で予想、を繰り返して実力を測る"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import COST_PIPS, HORIZON, PAIRS, THRESHOLD, pip_size
from .model import predict_proba, train


def walk_forward(f: pd.DataFrame, folds: int = 5, test_frac: float = 0.5) -> pd.DataFrame:
    f = f.dropna(subset=["target"])
    times = f.index.unique().sort_values()
    start = int(len(times) * (1 - test_frac))
    edges = np.linspace(start, len(times), folds + 1).astype(int)
    preds = []
    for a, b in zip(edges[:-1], edges[1:]):
        t0, t1 = times[a], times[b - 1]
        gap = times[max(a - HORIZON, 0)]  # 目的変数がテスト期間に重なる学習データを除く
        m = train(f[f.index < gap])
        te = f[(f.index >= t0) & (f.index <= t1)].copy()
        te["p_up"] = predict_proba(m, te).values
        preds.append(te)
    return pd.concat(preds)


def trades(p: pd.DataFrame, th: float = THRESHOLD) -> pd.DataFrame:
    """しきい値を超えたら HORIZON 本保有。同じペアで保有中は新規エントリーしない"""
    rows = []
    for t, g in p.groupby("ticker"):
        g = g.sort_index()
        busy_until = -1
        for i, (ts, r) in enumerate(g.iterrows()):
            if i <= busy_until:
                continue
            side = 1 if r.p_up >= th else -1 if r.p_up <= 1 - th else 0
            if side == 0:
                continue
            pips = side * (np.exp(r.fwd_ret) - 1) * r.close / pip_size(t) - COST_PIPS
            rows.append(dict(time=ts, pair=PAIRS[t], side=side, p_up=r.p_up, pips=pips))
            busy_until = i + HORIZON - 1
    return pd.DataFrame(rows)


def report(p: pd.DataFrame) -> str:
    y = p["target"].astype(int)
    lines = [f"検証期間: {p.index.min():%Y-%m-%d} 〜 {p.index.max():%Y-%m-%d} / {HORIZON}本先(={HORIZON*15}分後)の方向",
             f"全予想の正解率: {((p.p_up > .5) == y).mean():.1%}  AUC: {roc_auc_score(y, p.p_up):.3f}  "
             f"(上昇の割合={y.mean():.1%})"]
    for th in (0.55, 0.60, 0.65):
        tr = trades(p, th)
        if tr.empty:
            lines.append(f"しきい値{th:.2f}: シグナルなし"); continue
        lines.append(f"しきい値{th:.2f}: 取引{len(tr)}回 勝率{(tr.pips > 0).mean():.1%} "
                     f"平均{tr.pips.mean():+.2f}pips 合計{tr.pips.sum():+.1f}pips (コスト{COST_PIPS}pips込み)")
    lines.append("ペア別 正解率: " + ", ".join(
        f"{PAIRS[t]} {((g.p_up > .5) == g.target).mean():.1%}" for t, g in p.groupby("ticker")))
    return "\n".join(lines)
