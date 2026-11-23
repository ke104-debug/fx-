"""使い方:
  python -m fxpred backtest   # 過去データで予想の実力を検証して表示
  python -m fxpred run        # 最新データで予想し、シグナルを Discord に通知 (15分ごとに実行)
"""
import json
import os
import sys
from datetime import datetime, timezone

from . import backtest, data, features, model, notify
from .config import HORIZON, PAIRS, STATE_FILE, THRESHOLD


def cmd_backtest():
    f = features.build_all(data.fetch())
    print(backtest.report(backtest.walk_forward(f)))


def cmd_run():
    f = features.build_all(data.fetch())
    m = model.train(f)
    latest = f.groupby("ticker").tail(1)
    latest = latest.assign(p_up=model.predict_proba(m, latest).values)
    state = json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}
    now = datetime.now(timezone.utc)
    signals, table = [], []
    for ts, r in latest.iterrows():
        name = PAIRS[r.ticker]
        side = "BUY" if r.p_up >= THRESHOLD else "SELL" if r.p_up <= 1 - THRESHOLD else "-"
        conf = max(r.p_up, 1 - r.p_up)
        table.append(f"`{name}` {r.close:.5g}  上昇確率 {r.p_up:.0%}  {side}")
        prev = state.get(name, {})
        fresh = (prev.get("side") != side or
                 (now - datetime.fromisoformat(prev["time"])).total_seconds() > HORIZON * 900)
        if side != "-" and fresh:
            signals.append(dict(name=f"{'🟢 買い' if side == 'BUY' else '🔴 売り'} {name}",
                                value=f"価格 {r.close:.5g} / 確信度 {conf:.0%} / {HORIZON*15}分後予想 (足 {ts:%H:%M} UTC)",
                                inline=False))
            state[name] = {"side": side, "time": now.isoformat()}
        elif side == "-":
            state.pop(name, None)
    print("\n".join(table))
    if signals or os.getenv("FX_NOTIFY_ALL") == "1":
        notify.send(f"FX AIシグナル {now:%m/%d %H:%M} UTC",
                    "\n".join(table) + "\n\n※AIの予想であり利益を保証しません。最終判断はご自身で。",
                    color=5763719 if signals else 9807270, fields=signals)
    os.makedirs(os.path.dirname(STATE_FILE) or ".", exist_ok=True)
    json.dump(state, open(STATE_FILE, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    {"backtest": cmd_backtest, "run": cmd_run}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
