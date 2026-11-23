"""Dukascopy の公開データから USD/JPY の1時間足 (月ごとのファイル) をゆっくり取得して data/duka/ に保存する。
サーバーの負荷制限に配慮し、1ファイルごとに数秒あけ、429 などが返れば長めに待って再試行する。"""
import os, sys, time, urllib.request, urllib.error

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "duka")
os.makedirs(OUT, exist_ok=True)
years = range(int(sys.argv[1]) if len(sys.argv) > 1 else 2003, int(sys.argv[2]) if len(sys.argv) > 2 else 2025)
for side in ("BID",):
    for y in years:
        for m in range(12):
            path = os.path.join(OUT, f"USDJPY_{y}_{m + 1:02d}_{side}_hour.bi5")
            if os.path.exists(path):
                continue
            url = f"https://datafeed.dukascopy.com/datafeed/USDJPY/{y}/{m:02d}/{side}_candles_hour_1.bi5"
            for attempt in range(6):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    data = urllib.request.urlopen(req, timeout=30).read()
                    if data[:1] == b"{":
                        raise RuntimeError(data[:120])
                    open(path, "wb").write(data)
                    print("ok", y, m + 1, len(data), flush=True)
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        print("none", y, m + 1, flush=True); open(path, "wb").close(); break
                    wait = 30 * (attempt + 1)
                    print("retry", y, m + 1, e.code, f"wait {wait}s", flush=True); time.sleep(wait)
                except Exception as e:
                    wait = 30 * (attempt + 1)
                    print("retry", y, m + 1, str(e)[:100], f"wait {wait}s", flush=True)
                    time.sleep(wait)
            time.sleep(8)
print("done", flush=True)
