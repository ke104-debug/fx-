"""データ監視・システム保守部隊: 価格データの欠損・遅延・時刻ずれを見つける。
異常なら「新しく入らない」に切り替える (scan.py が合図を送らず、1日1回だけ通知する)。"""
import pandas as pd


def check(h: pd.DataFrame, now=None) -> dict:
    now = now or pd.Timestamp.now(tz="UTC")
    probs = []
    last = h.index[-1]
    jst = now + pd.Timedelta(hours=9)
    market_open = not (jst.weekday() == 5 and jst.hour >= 7) and not (jst.weekday() == 6) and not (jst.weekday() == 0 and jst.hour < 7)
    if last > now + pd.Timedelta(minutes=5):
        probs.append(f"最新の足の時刻が未来 ({last:%m/%d %H:%M} UTC)。時刻がずれている")
    if market_open and now - last > pd.Timedelta(hours=3):
        probs.append(f"価格が{(now - last).total_seconds() / 3600:.0f}時間更新されていない")
    w = h.iloc[-48:]
    gaps = w.index.to_series().diff().dt.total_seconds().div(3600)
    # 週末を除いた2時間以上の抜け
    gaps = gaps[(gaps > 2) & ~((w.index.weekday == 6) | (w.index.weekday == 0))]
    if len(gaps):
        probs.append(f"直近48本で足の抜けが{len(gaps)}か所 (最大{gaps.max():.0f}時間)")
    flat = (w.High == w.Low).sum()
    if flat >= 6:
        probs.append(f"値が動いていない足が{flat}本 (データ元の不具合の可能性)")
    c = h.Close.iloc[-200:]
    jump = (c.pct_change().abs() > 0.03).sum()
    if jump:
        probs.append(f"1時間で3%を超える飛び値が{jump}回 (データの誤りの可能性)")
    return {"ok": not probs, "problems": probs}
