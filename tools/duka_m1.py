"""Dukascopy の公開データから 1分足を日ごとに取得し、1つの npz (時刻・始値・高値・安値・終値) にまとめる。
使い方: python tools/duka_m1.py 2004-01-01 2006-12-31 out.npz [制限秒] [取得済みの日のリスト] [通貨ペア]
取得済みの日のリスト (1行1日 YYYY-MM-DD) にある日は飛ばす。通貨ペアの既定は USDJPY
接続制限があるので1リクエストごとに数秒あけ、失敗した日は後回しにして何周かする。土曜は取引がないので飛ばす。"""
import lzma, os, random, sys, time, urllib.error, urllib.request
from datetime import date, datetime, timedelta, timezone

import numpy as np

d0, d1, out = date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2]), sys.argv[3]
limit = float(sys.argv[4]) if len(sys.argv) > 4 else 15000   # 秒。これを過ぎたら取れた分だけ保存
days = [d0 + timedelta(n) for n in range((d1 - d0).days + 1)]
PAIR = sys.argv[6] if len(sys.argv) > 6 else "USDJPY"
SCALE = 1000 if PAIR.endswith("JPY") else 100000
have = set()
if len(sys.argv) > 5 and os.path.exists(sys.argv[5]):
    have = {date.fromisoformat(x.strip()) for x in open(sys.argv[5]) if x.strip()}
days = [d for d in days if d.weekday() != 5 and d <= date.today() and d not in have]
got: dict = {}
t_start = time.time()
for rnd in range(40):
    todo = [d for d in days if d not in got]
    print(f"pass {rnd}: {len(todo)} days left", flush=True)
    if not todo or time.time() - t_start > limit:
        break
    random.shuffle(todo)
    for d in todo:
        if time.time() - t_start > limit:
            break
        url = f"https://datafeed.dukascopy.com/datafeed/{PAIR}/{d.year}/{d.month - 1:02d}/{d.day:02d}/BID_candles_min_1.bi5"
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read()
            if raw[:1] == b"{":
                raise RuntimeError("rate limited")
            if not raw:
                got[d] = None
            else:
                b = np.frombuffer(lzma.decompress(raw), dtype=">i4").reshape(-1, 6)
                base = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
                got[d] = (b[:, 0].astype(np.int64) + base, b[:, 1:5].astype(np.float64) / SCALE)   # open, close, low, high
        except urllib.error.HTTPError as e:
            if e.code == 404:
                got[d] = None
            else:
                print("skip", d, e.code, flush=True); time.sleep(20)
        except Exception as e:
            print("skip", d, str(e)[:60], flush=True); time.sleep(20)
        time.sleep(3)
parts = [v for k, v in sorted(got.items()) if v is not None]
t = np.concatenate([p[0] for p in parts]) if parts else np.array([], np.int64)
px = np.concatenate([p[1] for p in parts]) if parts else np.zeros((0, 4))
# 取引のない時間は同じ値で埋まっている (出来高0)。始値=高値=安値=終値 の足も残す (後で必要なら除く)
np.savez_compressed(out, t=t, o=px[:, 0].astype(np.float32), c=px[:, 1].astype(np.float32),
                    l=px[:, 2].astype(np.float32), h=px[:, 3].astype(np.float32))
print(f"saved {out}: {len(got)}/{len(days)} days, {len(t)} bars", flush=True)
