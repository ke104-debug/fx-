"""シミュレーション・検証部隊の週次の再検証: 改善研究部隊の進言AIが挙げた候補を、記録で検証して重みを決め直す。
- 重み0なのに当たっている担当: 実際の記録で 80回以上・的中55%超・平均+3pips超、かつ偶然では説明しにくい (符号検定 p<0.05) なら
  小さな重み (0.3) を与える。23年の履歴がある担当は、次に Claude が23年で再検証するまで 0.3 を上限にする
- 重みがあるのに外れている担当: 実際の記録で 100回以上・平均がマイナスなら重みを半分に (2回続けば0)
- 決めたことは journal/org_decisions.csv に残す (監査AIが変えすぎを見張る)"""
import json
import math
import os

import pandas as pd

from .advisor import live_scores

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
W_PATH = os.path.join(ROOT, "config", "org_weights.json")
DEC = os.path.join(ROOT, "journal", "org_decisions.csv")


def _sign_p(n, hit):
    """的中率 hit が偶然 (50%) で出る確率 (片側、正規近似)"""
    if n <= 0:
        return 1.0
    z = (hit - 0.5) * math.sqrt(n) / 0.5
    return 0.5 * math.erfc(z / math.sqrt(2))


def run(apply=True) -> list[str]:
    s = live_scores()
    if s.empty:
        return ["再検証: 記録がまだ少ないため見送り"]
    allw = json.load(open(W_PATH))
    W = allw.setdefault("weights", {})
    msgs, rows = [], []
    now = pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d")
    for r in s.itertuples():
        if r.name.startswith("統括AI"):
            continue
        w0 = W.get(r.name, 0.0)
        p = _sign_p(r.n, r.hit)
        new, why = w0, ""
        if w0 == 0 and r.n >= 80 and r.hit > 0.55 and r.pips > 3 and p < 0.05:
            new, why = 0.3, f"重み0だが記録で当たっている ({r.n}回・的中{r.hit:.0%}・平均{r.pips:+.1f}pips・偶然の確率{p:.1%})"
        elif w0 > 0 and r.n >= 100 and r.pips < 0:
            new = 0.0 if w0 <= 0.3 else round(w0 / 2, 2)
            why = f"重みがあるのに記録で外れている ({r.n}回・平均{r.pips:+.1f}pips)"
        if new != w0:
            msgs.append(f"「{r.name.split('|')[-1]}」重み {w0} → {new}: {why}")
            rows.append(dict(date=now, name=r.name, before=w0, after=new, reason=why))
            W[r.name] = new
    if rows and apply:
        json.dump(allw, open(W_PATH, "w"), ensure_ascii=False, indent=1)
        pd.DataFrame(rows).to_csv(DEC, mode="a", header=not os.path.exists(DEC), index=False)
    return msgs or ["再検証: 重みを変える担当はなし"]
