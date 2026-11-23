"""相場分析AIチーム (追加料金なし: すべて手元の機械学習で動く)

  相場判定AI   : 今の相場を「トレンド / 通常(レンジ) / 急変」に分ける (regime.label)
  担当AI x3    : 各相場で使える作戦の候補シグナルを、その相場の過去の取引だけで学習した LightGBM が採点する
                 (予想される損益 R を出す。R = 損益 ÷ 損切り幅)
  戦略選択AI   : 担当AIの採点がプラスのシグナルだけを選び、採点の高さに応じて1回のリスクを決める

検証は「その年より前のデータだけで学習し、その年を運用する」をくり返す (ウォークフォワード)。
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import portfolio as pf
from . import regime as rg
from . import shock as sh
from . import strategies as st

# 各担当AIが扱う作戦の候補
CANDIDATES = {
    "トレンド": ["H1 ブレイクアウト(5日高安値)", "H2 押し目買い/戻り売り", "H3 東京レンジのロンドン抜け"],
    "通常": ["H3 東京レンジのロンドン抜け", "H4 レンジ逆張り+流れフィルタ", "H5 指標発表直後の順張り", "D ボリンジャー逆張り"],
    "荒れ": ["S 急変の反動(流れの方向)", "S2 急変の反動(流れと逆)", "H5 指標発表直後の順張り", "H1 ブレイクアウト(5日高安値)"],
}
MAX_RISK = {"トレンド": 0.015, "通常": 0.015, "荒れ": 0.04}
MIN_TRAIN = 120          # 学習に必要な最低取引数 (足りない担当AIは見送り)
FEATURES = ["strategy_id", "side", "hour", "weekday", "atr_ratio", "adx", "rsi", "dist_ema200", "slope20d",
            "width_atr", "stop_atr", "rng24_atr", "ret24", "ret120"]
STRATS = sorted({s for v in CANDIDATES.values() for s in v})


def _bollinger(d, i):
    s = st.bollinger_reversion(d, i)
    return (s[0], s[1], s[2], None) if s else None


def candidate_trades(h: pd.DataFrame) -> pd.DataFrame:
    """全作戦の候補取引 (損益 R つき) を集める"""
    rules = dict(pf.HOURLY); rules["D ボリンジャー逆張り"] = _bollinger
    out = []
    for name in STRATS:
        if name.startswith("S"):
            continue
        _, tr = pf.run_hourly(h, rules[name], diff=None)
        if len(tr):
            out.append(tr[["entry", "exit", "side", "R", "dist", "price"]].assign(作戦=name))
    up = h.Close.ewm(span=480).mean().diff(24) > 0
    for side in (1, -1):
        t = sh.find(h, side=side)
        if t.empty:
            continue
        aligned = up.reindex(t.検知).values == (side == 1)
        t = pd.DataFrame({"entry": t.入り, "exit": t.入り, "side": side, "R": t.R, "dist": t.dist, "price": t.price,
                          "作戦": np.where(aligned, "S 急変の反動(流れの方向)", "S2 急変の反動(流れと逆)")})
        out.append(t)
    a = pd.concat(out).sort_values("entry").reset_index(drop=True)
    return add_features(h, a)


def add_features(h: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    """入る直前の確定足の情報だけを特徴量にする (未来を見ない)"""
    f = pd.DataFrame(index=h.index)
    f["atr_ratio"] = h.atr / h.atr.rolling(480).mean()
    f["adx"] = h.adx
    f["rsi"] = h.rsi
    f["dist_ema200"] = (h.Close - h.ema200) / h.atr
    ema = h.Close.ewm(span=480).mean()
    f["slope20d"] = (ema / ema.shift(120) - 1) * 100
    f["width_atr"] = h.width / h.atr
    f["rng24_atr"] = (h.High.rolling(24).max() - h.Low.rolling(24).min()) / h.atr
    f["ret24"] = (h.Close / h.Close.shift(24) - 1) * 100
    f["ret120"] = (h.Close / h.Close.shift(120) - 1) * 100
    f["regime"] = h.regime.map(rg.NAMES)
    prev = a.entry - pd.Timedelta(hours=1)
    pos = f.index.searchsorted(prev, side="right") - 1
    pos = np.clip(pos, 0, len(f) - 1)
    x = f.iloc[pos].reset_index(drop=True)
    a = pd.concat([a.reset_index(drop=True), x], axis=1)
    a["状態"] = a.pop("regime")
    a["strategy_id"] = a.作戦.map({s: i for i, s in enumerate(STRATS)})
    a["hour"] = a.entry.dt.hour
    a["weekday"] = a.entry.dt.weekday
    a["stop_atr"] = a.dist / h.atr.reindex(a.entry - pd.Timedelta(hours=1), method="ffill").values
    # 担当AIの割り当て: その相場状態の候補に入っている作戦だけ
    a = a[[s in CANDIDATES.get(r, []) for s, r in zip(a.作戦, a.状態)]]
    return a.reset_index(drop=True)


def train_specialist(train: pd.DataFrame):
    if len(train) < MIN_TRAIN:
        return None
    m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=30,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=0)
    m.fit(train[FEATURES], train.R.clip(-1.5, 4), categorical_feature=["strategy_id"])
    return m


def selector_risk(pred: float, regime: str) -> float:
    """戦略選択AI: 採点 (予想R) がプラスの時だけ、高いほどリスクを大きく (上限あり)"""
    if pred <= 0.05:
        return 0.0
    return min(MAX_RISK[regime], MAX_RISK[regime] * pred / 0.4)


def walk_forward(a: pd.DataFrame, first_test_year: int, train_years: int = 8) -> pd.DataFrame:
    rows = []
    for y in range(first_test_year, a.entry.dt.year.max() + 1):
        start = pd.Timestamp(f"{y}-01-01", tz="UTC")
        test = a[(a.entry >= start) & (a.entry < pd.Timestamp(f"{y + 1}-01-01", tz="UTC"))]
        for reg in CANDIDATES:
            tr = a[(a.exit < start) & (a.entry >= start - pd.DateOffset(years=train_years)) & (a.状態 == reg)]
            te = test[test.状態 == reg].copy()
            if te.empty:
                continue
            m = train_specialist(tr)
            te["予想R"] = m.predict(te[FEATURES]) if m is not None else np.nan
            te["学習数"] = len(tr)
            rows.append(te)
    return pd.concat(rows).sort_values("entry").reset_index(drop=True) if rows else pd.DataFrame()


def simulate(p: pd.DataFrame, mode: str, init=1_000_000) -> tuple[pd.Series, pd.DataFrame]:
    """mode: 'ai' (担当AI+戦略選択AI) / 'all' (候補を全部、固定リスク) / 'rules' (scan.RULES の固定ルール)"""
    from .scan import RULES
    key = {"S": "S 急変の反動(流れの方向)", "H3": "H3 東京レンジのロンドン抜け", "H4": "H4 レンジ逆張り+流れフィルタ", "H5": "H5 指標発表直後の順張り"}
    fixed = {(key[k], s): r for (k, s), r in RULES.items()}
    eq, path, taken = init, [], []
    for r in p.itertuples():
        if mode == "ai":
            f = selector_risk(r.予想R, r.状態) if not np.isnan(r.予想R) else 0.0
        elif mode == "all":
            f = 0.01
        else:
            f = fixed.get((r.作戦, r.状態), 0.0)
        if f <= 0:
            continue
        eq *= 1 + min(f, pf.MAX_LEV * r.dist / r.price) * r.R
        path.append((r.entry, eq)); taken.append(r)
    return pd.Series(dict(path), dtype=float), pd.DataFrame(taken)


def report(p: pd.DataFrame):
    out = []
    for mode, label in (("ai", "AIチーム (担当AI+戦略選択AI)"), ("rules", "固定ルール (今の監視設定)"), ("all", "候補を全部 (各1%)")):
        cv, t = simulate(p, mode)
        if cv.empty:
            out.append({"運用": label, "取引": 0}); continue
        cv.index = pd.DatetimeIndex(cv.index)
        y = cv.resample("YE").last().ffill()
        yr = (y / y.shift(1).fillna(1e6) - 1)
        out.append({"運用": label, "最終(万円)": round(cv.iloc[-1] / 1e4), "最大下落": f"{(1 - cv / cv.cummax()).max():.0%}",
                    "取引": len(t), "勝率": f"{(t.R > 0).mean():.0%}", "プラスの年": f"{(yr > 0).sum()}/{len(yr)}",
                    "年別%": {k.year: int(round(v * 100)) for k, v in yr.items()}})
    pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 200)
    print(pd.DataFrame(out).to_string(index=False))
    sel = p[p.予想R.notna()]
    if len(sel):
        sel = sel.assign(採点=pd.cut(sel.予想R, [-9, 0, 0.05, 0.2, 9], labels=["マイナス", "0〜0.05", "0.05〜0.2", "0.2以上"]))
        print("\n担当AIの採点ごとの実際の平均R (採点が高いほど実際も良ければ、AIが学習できている):")
        print(sel.groupby(["状態", "採点"], observed=True).R.agg(["mean", "size"]).round(2).unstack("採点").to_string())
