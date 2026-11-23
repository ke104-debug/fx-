"""通知のテスト: 過去の場面を再現して「チャンスの通知」と「反省の報告」を [テスト] 付きで送る。
運用の記録 (journal/・state/) には一切書き込まない。"""
import json
import os
import tempfile

import pandas as pd

from . import dailyplan as dp, notify, regime as rg, scan

_send = notify.send


def tagged(title, body, **kw):
    return _send("[テスト] " + title, body, **kw)


def run(dry=False):
    tmp = tempfile.mkdtemp()
    notify.send = (lambda t, b, **k: print("[テスト] " + t + "\n" + b)) if dry else tagged
    full = scan.load_recent()

    # 1) チャンスの通知: 11/23 17:00 (日本時間) の足が確定した時点を再現
    state = os.path.join(tmp, "scan.json")
    json.dump({"regime": "通常"}, open(state, "w"))
    scan.STATE = state
    scan.log_signal = lambda *a, **k: None
    scan.load_recent = lambda: full.loc[:"2026-11-23 08:00"]
    scan.run()

    # 2) 反省の報告: 11/20 朝10時 (日本時間) の予想を、その後24時間の実際の値動きで答え合わせ
    dp.J = tmp
    dp.FORECAST, dp.REVIEW, dp.LEARN = (os.path.join(tmp, f) for f in ("f.csv", "r.csv", "l.json"))
    h = rg.prepare(full)
    for t in pd.date_range("2026-10-26 01:00", "2026-11-20 01:00", freq="D", tz="UTC"):
        sub = h.loc[:t - pd.Timedelta(hours=1)]
        learn = dp.load_learning()
        dp.forecast(sub, dp.analyze(sub, learn), learn)
        if t < pd.Timestamp("2026-11-20 01:00", tz="UTC"):
            dp.review(sub, dp.load_learning())
    reviews = dp.review(h, dp.load_learning())   # 金曜の予想は週明けの足までの24本で答え合わせ
    learn = dp.load_learning()
    a = dp.analyze(h, learn)
    fc = dp.forecast(h, a, learn)
    title, body = dp.morning_message(h, a, fc, reviews[-1:], learn)
    note = ("(テスト用の再現: 11/20 朝10時の予想を、その後24時間の実際の値動きで答え合わせしたもの。"
            "過去の日の原因推定には今日時点のニュースを使っています)\n\n")
    notify.send(title, note + body, color=3447003)
