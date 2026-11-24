"""改善研究部隊の提案の手続き (暴走・自己正当化を防ぐ)。
1) 新しいルールは必ずここに「提案」として登録する (誰が・何を・どの期間で作ったか)
2) シミュレーション・検証部隊が、作るのに使っていない期間 (holdout) で確かめ、今のルールと比べる (compared)
3) 両方を満たしたものだけ「候補」→ 影の運用 or 利用者の承認 → 「採用」。満たさなければ「却下」
監査AIは、holdout の記録が無いまま「採用」になったルールを指摘する。記録: journal/proposals.csv"""
import os

import pandas as pd

PATH = os.path.join(os.path.dirname(__file__), "..", "..", "journal", "proposals.csv")
COLS = ["id", "date", "by", "title", "built_on", "holdout", "compared", "status", "note"]


def load() -> pd.DataFrame:
    return pd.read_csv(PATH) if os.path.exists(PATH) else pd.DataFrame(columns=COLS)


def add(by, title, built_on, holdout="", compared="", status="提案", note="") -> str:
    d = load()
    pid = f"P{len(d) + 1:03d}"
    row = dict(id=pid, date=f"{pd.Timestamp.now(tz='Asia/Tokyo'):%Y-%m-%d}", by=by, title=title, built_on=built_on,
               holdout=holdout, compared=compared, status=status, note=note)
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    pd.concat([d, pd.DataFrame([row])]).to_csv(PATH, index=False)
    return pid


def update(pid, **kw):
    d = load()
    for k, v in kw.items():
        d.loc[d.id == pid, k] = v
    d.to_csv(PATH, index=False)
