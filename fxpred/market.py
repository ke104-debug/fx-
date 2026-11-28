"""為替市場が開いているか (時計役 worker/src.js と同じ決まり)。
日曜 20:00 UTC 〜 金曜 22:00 UTC を『開いている』とする (夏冬の1時間のずれを含めて広めに取る)。
`python -m fxpred.market` は開いていれば 0、閉まっていれば 1 で終わる (Actions で、閉まっている間は見張りを始めないために使う)。
標準ライブラリだけで動く (部品を入れる前に判断できる)"""
import sys
from datetime import datetime, timezone


def is_open(d: datetime | None = None) -> bool:
    d = d or datetime.now(timezone.utc)
    wd, h = d.weekday(), d.hour          # 月=0 … 土=5, 日=6
    if wd == 5:
        return False
    if wd == 6 and h < 20:
        return False
    if wd == 4 and h >= 22:
        return False
    return True


if __name__ == "__main__":
    o = is_open()
    print("市場は開いています" if o else "市場が閉まっているため、見張り・分析をしません (週末の無駄な実行を止める)")
    sys.exit(0 if o else 1)
