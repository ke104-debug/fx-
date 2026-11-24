"""Dukascopy の公開データから USD/JPY (第3引数で EURUSD なども可) の1時間足 (月ごとのファイル) をゆっくり取得して data/duka/ に保存する。
接続が切られやすいので、失敗した月は後回しにして全体を何周かして埋める。1リクエストごとに10秒あける。"""
import os, random, sys, time, urllib.error, urllib.request

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "duka")
os.makedirs(OUT, exist_ok=True)
y0, y1 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2004, 2025)
PAIR = sys.argv[3] if len(sys.argv) > 3 else "USDJPY"
months = [(y, m) for y in range(y0, y1) for m in range(12)]
for rnd in range(30):
    todo = [(y, m) for y, m in months if not os.path.exists(os.path.join(OUT, f"{PAIR}_{y}_{m + 1:02d}_BID_hour.bi5"))]
    print(f"pass {rnd}: {len(todo)} files left", flush=True)
    if not todo:
        break
    random.shuffle(todo)
    for y, m in todo:
        path = os.path.join(OUT, f"{PAIR}_{y}_{m + 1:02d}_BID_hour.bi5")
        url = f"https://datafeed.dukascopy.com/datafeed/{PAIR}/{y}/{m:02d}/BID_candles_hour_1.bi5"
        try:
            data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read()
            if data[:1] == b"{":
                raise RuntimeError("rate limited")
            open(path, "wb").write(data)
            print("ok", y, m + 1, len(data), flush=True)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                open(path, "wb").close(); print("none", y, m + 1, flush=True)
            else:
                print("skip", y, m + 1, e.code, flush=True)
        except Exception as e:
            print("skip", y, m + 1, str(e)[:60], flush=True)
        time.sleep(10)
    time.sleep(60)
print("done", flush=True)
