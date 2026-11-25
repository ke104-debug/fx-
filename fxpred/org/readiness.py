"""自動売買への準備度 (利用者の方針: チャンス通知が成果を挙げていれば、少額で自動売買を試したい)。
進む条件を先に数字で決めておき、毎週の見直しで採点する。全部そろうまで自動売買には進まない (進む時も利用者の承認を得る)。

段階: ① 通知の成績を確かめる (ルール凍結後) → ② デモ口座で自動発注を試す (2週間) → ③ 少額の本番
① の条件 (ルール凍結 2026-11-25 以降の記録だけで数える):
  - 期間 4週間以上、送った合図 30回以上 (答えが出たもの)
  - 合計がプラス、総利益/総損失 1.2以上、1回平均 +0.1R以上
  - 最大の連敗が 10回以内、合計の最大の落ち込みが -8R 以内
  - 運用の安定: 直近2週間の監視の実行がそろっている (朝の通知の稼働の行) ・止まる不具合が無い"""
import os

import numpy as np
import pandas as pd

FREEZE = pd.Timestamp("2026-11-25", tz="UTC")


def check(h: pd.DataFrame) -> list[str]:
    from ..periodic import evaluate_signals
    ev = evaluate_signals(h)
    if ev.empty:
        return ["自動売買への準備度: 送った合図の記録がまだない (条件: 4週間以上・30回以上)"]
    ev["t"] = pd.to_datetime(ev.time_utc, utc=True)
    s = ev[(ev.t >= FREEZE) & (ev.sent.astype(str) == "True") & ev.pips.notna()].sort_values("t").copy()
    weeks = (pd.Timestamp.now(tz="UTC") - FREEZE).days / 7
    risk = (s.entry.astype(float) - s.stop.astype(float)).abs() * 100
    s["R"] = s.pips / risk.replace(0, np.nan)
    n = len(s)
    win, loss = s.R[s.R > 0].sum(), -s.R[s.R < 0].sum()
    pf = win / loss if loss > 0 else (np.inf if win > 0 else 0)
    streak = mx = 0
    for r in s.R:
        streak = streak + 1 if r < 0 else 0
        mx = max(mx, streak)
    eq = s.R.cumsum()
    dd = float((eq - eq.cummax()).min()) if n else 0.0
    gates = [("期間 4週間以上", weeks >= 4, f"{weeks:.1f}週"),
             ("送った合図 30回以上", n >= 30, f"{n}回"),
             ("合計がプラス", n > 0 and s.R.sum() > 0, f"{s.R.sum():+.1f}R" if n else "-"),
             ("総利益/総損失 1.2以上", pf >= 1.2, f"{pf:.2f}" if n else "-"),
             ("1回平均 +0.1R以上", n > 0 and s.R.mean() >= 0.1, f"{s.R.mean():+.2f}R" if n else "-"),
             ("最大の連敗 10回以内", mx <= 10, f"{mx}回"),
             ("最大の落ち込み -8R以内", dd >= -8, f"{dd:+.1f}R")]
    ok = sum(g[1] for g in gates)
    out = [f"自動売買への準備度 (段階①: 通知の成績): {ok}/{len(gates)} 条件クリア" + (" → 段階② (デモ口座) を相談できる" if ok == len(gates) else "")]
    out += [f"{'○' if g else '×'} {name}: {val}" for name, g, val in gates]
    return out
