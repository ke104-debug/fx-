"""手数料 (スプレッド) の見積もり (利用者の指摘 K001, 2026-11-27)。
SBI 証券のドル円は普段 往復約0.2pips。時間帯で広がる分は、Dukascopy の売値と買値の差 (1時間足の終値) の時間帯ごとの中央値から、
普段 (日本時間 10時〜翌1時) に対する倍率で見積もる。2023年1〜4月の実測: 普段 約0.8pips に対し 日本時間5時 3.0 / 6時 4.7 / 7時 3.9 (約4〜6倍)。
→ SBI でも同じ時間帯は広がると見て 0.2 × 倍率。重要指標の発表直後 (5分以内) は実測がまだ無いので 5倍と仮定 (1分足・ティックで実測して置き換える)。
全期間の実測は GitHub Actions (duka-spread.yml) で取り、config/spread_by_hour.json を作り直す。"""
import json
import os

import pandas as pd

BASE = 0.2          # SBI の普段の往復スプレッド (pips)
PATH = os.path.join(os.path.dirname(__file__), "..", "config", "spread_by_hour.json")
# 日本時間の時刻 → 普段に対する倍率 (2023年1〜4月の Dukascopy 実測の中央値 / 普段0.8)。ファイルがあればそちらを使う
DEFAULT = {h: 1.0 for h in range(24)}
DEFAULT.update({5: 3.7, 6: 5.9, 7: 4.9, 8: 1.25, 9: 1.25})
EVENT_X = 5.0       # 重要指標の発表から5分以内 (仮定)
MAX_SHARE = 0.10    # 手数料が損切り幅の1割を超えたら注意、2割で見送り


def table() -> dict:
    try:
        return {int(k): float(v) for k, v in json.load(open(PATH))["factor"].items()}
    except Exception:
        return DEFAULT


def pips(t: pd.Timestamp | None = None, event_near=False) -> float:
    """その時刻の往復手数料の見積もり (pips)"""
    t = t or pd.Timestamp.now(tz="UTC")
    j = t + pd.Timedelta(hours=9)
    x = table().get(j.hour, 1.0)
    if j.weekday() == 0 and j.hour < 9:      # 月曜の朝は週末明けで広がりやすい
        x = max(x, 3.0)
    if event_near:
        x = max(x, EVENT_X)
    return round(BASE * x, 2)


def check(entry: float, stop: float, t=None, event_near=False) -> tuple[str, str]:
    """('ok'|'注意'|'見送り', 説明)。手数料が損切り幅の何%か"""
    c = pips(t, event_near)
    risk = abs(entry - stop) * 100
    share = c / max(risk, 1e-9)
    txt = f"手数料の目安 {c:.1f}pips (損切り幅 {risk:.0f}pips の {share:.0%})"
    if share >= 2 * MAX_SHARE:
        return "見送り", txt + " → 手数料が高すぎる場面"
    if share >= MAX_SHARE:
        return "注意", txt + " → 手数料が高め"
    return "ok", txt
