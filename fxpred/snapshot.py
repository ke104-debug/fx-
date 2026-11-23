"""Web ページ用に USD/JPY の「テクニカル予想」と「システム状況」を1つの JSON にまとめる"""
import os
from datetime import datetime, timezone

import pandas as pd

from . import backtest, data, diagnose, features as ft, model
from .config import PAIRS

TICKER, NAME = "USDJPY=X", "USD/JPY"


def _model_probs(d) -> dict:
    """1時間後(4本)・4時間後(16本)の上昇確率と、直近の検証正解率"""
    out = {}
    usd = ft.usd_strength(d)
    for label, h in (("1時間後", 4), ("4時間後", 16)):
        f = pd.concat([ft.build(df, t, usd, horizon=h) for t, df in d.items()]).sort_index()
        m = model.train(f)
        last = f[f.ticker == TICKER].tail(1)
        p = float(model.predict_proba(m, last).iloc[0])
        wf = backtest.walk_forward(f, folds=3)
        wf = wf[wf.ticker == TICKER]
        acc = float(((wf.p_up > .5) == wf.target).mean())
        out[label] = {"上昇確率": round(p, 3), "予想": "上昇" if p >= .55 else "下落" if p <= .45 else "中立",
                      "直近検証の正解率": round(acc, 3), "検証件数": int(len(wf))}
    return out


def _journal_stats() -> dict:
    if not os.path.exists(diagnose.JOURNAL):
        return {"診断件数": 0}
    text = diagnose.evaluate()
    j = pd.read_csv(diagnose.JOURNAL)
    first = text.splitlines()[0]
    return {"診断件数": int(len(j)), "答え合わせ": first}


def run() -> dict:
    d = data.fetch()
    df = d[TICKER]
    tech = diagnose.technical(df)
    score = tech["テクニカルスコア"]
    probs = _model_probs(d)
    now = datetime.now(timezone.utc)
    last_bar = df.index[-1]
    lag_min = (now - last_bar.to_pydatetime()).total_seconds() / 60
    return {
        "generated_utc": now.strftime("%Y-%m-%d %H:%M"),
        "pair": NAME,
        "price": tech["価格"],
        "technical": {
            "総合": "上昇寄り" if score >= 2 else "下落寄り" if score <= -2 else "中立",
            "スコア": score,
            "時間足": {k: tech[k] for k in ("15分足", "1時間足", "4時間足")},
            "根拠": tech["テクニカル根拠"],
            "ボリンジャー位置(σ)": tech["ボリンジャー位置(σ)"],
            "24時間 高値/安値": tech["24時間 高値/安値"],
            "5日 高値/安値": tech["5日 高値/安値"],
            "4時間の平均変動幅": tech["4時間の平均変動幅(目安)"],
            "AIモデル": probs,
        },
        "macro": diagnose.macro_snapshot(),
        "system": {
            "データ取得": "正常" if lag_min < 60 else f"遅延 (最新足から{lag_min:.0f}分)",
            "最新足_utc": last_bar.strftime("%Y-%m-%d %H:%M"),
            "データ件数": int(len(df)),
            "長期検証": "2012〜2026年の日足で正解率50〜51% (テクニカル・マクロとも偶然と区別できない)",
            **_journal_stats(),
        },
    }
