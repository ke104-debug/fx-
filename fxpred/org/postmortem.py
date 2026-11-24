"""原因究明部隊: 成績が悪い時に、どこで負けているかを切り分ける (毎週の見直しに載せる)。
送った合図の直近8週間の結果を、作戦・時間帯・曜日・相場の状態・重要指標の近さ・決済の仕方・方向の会議との対立 で分けて、
負けが集中している所を挙げる。少ない回数の偶然を拾わないよう、5回以上の区分だけを見る。"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def run(h: pd.DataFrame, weeks=8) -> list[str]:
    from ..periodic import evaluate_signals
    ev = evaluate_signals(h)
    if ev.empty:
        return ["原因究明: 送った合図の結果がまだない"]
    ev["t"] = pd.to_datetime(ev.time_utc, utc=True)
    ev = ev[(ev.sent.astype(str) == "True") & ev.pips.notna() & (ev.t >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(weeks=weeks))]
    if len(ev) < 5:
        return [f"原因究明: 直近{weeks}週の結果が{len(ev)}回で少なく、切り分けられない"]
    jst = ev.t + pd.Timedelta(hours=9)
    ev["時間帯"] = pd.cut(jst.dt.hour, [-1, 8, 15, 21, 24], labels=["早朝", "東京", "ロンドン", "NY"]).astype(str)
    ev["曜日"] = jst.dt.weekday.map(dict(enumerate("月火水木金土日")))
    try:
        reg = h.regime.reindex(ev.t.dt.floor("h"), method="ffill").values
        ev["相場の状態"] = pd.Series(reg, index=ev.index).map({"normal": "通常", "trend": "トレンド", "shock": "荒れ"}).fillna("?")
    except Exception:
        ev["相場の状態"] = "?"
    ev["作戦"] = ev.strategy.astype(str)
    ev["決済"] = ev.result.astype(str)
    try:
        org = pd.read_csv(os.path.join(ROOT, "journal", "org.csv"))
        org["t"] = pd.to_datetime(org.time_utc, utc=True)
        o = org.set_index("t").overall.sort_index()
        ov = o.reindex(ev.t.dt.floor("h"), method="ffill").values
        ev["会議との関係"] = np.where(np.sign(ov) * ev.side.astype(int) < 0, "逆向き", "同じ・中立")
    except Exception:
        ev["会議との関係"] = "?"
    tot = ev.pips.sum()
    out = [f"原因究明 (直近{weeks}週・{len(ev)}回・合計{tot:+.0f}pips)"]
    finds = []
    for col in ("作戦", "時間帯", "曜日", "相場の状態", "決済", "会議との関係"):
        for k, g in ev.groupby(col):
            if len(g) >= 5 and g.pips.sum() < 0:
                finds.append((g.pips.sum(), f"{col}「{k}」で負けが集中: {len(g)}回・合計{g.pips.sum():+.0f}pips・勝率{(g.pips > 0).mean():.0%}"))
    for _, txt in sorted(finds)[:4]:
        out.append(txt)
    if len(out) == 1:
        out.append("負けが集中している区分は見つからない (全体のぶれの範囲)")
    out.append("※ここで見つけた区分を止める案は、改善研究部隊の提案として登録し、23年のデータで確かめてから判断する")
    return out
