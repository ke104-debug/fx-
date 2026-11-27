"""Dukascopy の売値・買値の1時間足 (月ごとのファイル) から、日本時間の時刻ごとのスプレッドの中央値を求め、
普段 (日本時間 10時〜翌1時) に対する倍率を config/spread_by_hour.json に書く (fxpred/costs.py が使う)。GitHub Actions で動かす"""
import json, lzma, sys, time, urllib.request
import numpy as np, pandas as pd


def get(url):
    for k in range(8):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read()
        except Exception:
            time.sleep(5 * (k + 1))
    return None


y0, y1 = int(sys.argv[1]), int(sys.argv[2])
rows = []
now = pd.Timestamp.now(tz="UTC")
for y in range(y0, y1 + 1):
    for m in range(1, 13):
        if pd.Timestamp(year=y, month=m, day=1, tz="UTC") >= now.normalize().replace(day=1):
            break
        rec = {}
        for side in ("BID", "ASK"):
            raw = get(f"https://datafeed.dukascopy.com/datafeed/USDJPY/{y}/{m - 1:02d}/{side}_candles_hour_1.bi5")
            time.sleep(3)
            try:
                r = np.frombuffer(lzma.decompress(raw), dtype=">i4,>i4,>i4,>i4,>i4,>f4")
            except Exception:
                rec = {}; break
            rec[side] = pd.Series(r["f2"] / 1000, index=pd.Timestamp(year=y, month=m, day=1, tz="UTC") + pd.to_timedelta(r["f0"], unit="s"))
        if rec:
            v = pd.Series(r["f5"], index=rec["BID"].index) if False else None
            rows.append((rec["ASK"] - rec["BID"]).dropna() * 100)
            print(y, m, "ok", flush=True)
s = pd.concat(rows)
s = s[(s > 0) & (s < 8)]                    # 週末の埋め草 (10pips 前後の固定値) を除く
j = s.index + pd.Timedelta(hours=9)
s = s[j.weekday < 5]
j = s.index + pd.Timedelta(hours=9)
med = s.groupby(j.hour).median()
base = s[(j.hour >= 10) | (j.hour < 1)].median()
out = {"source": f"Dukascopy 1時間足の終値の売値−買値 {y0}-{y1}・平日", "base_dukascopy_pips": round(float(base), 3),
       "factor": {int(h): round(max(1.0, float(v / base)), 2) for h, v in med.items()},
       "median_pips": {int(h): round(float(v), 2) for h, v in med.items()}}
json.dump(out, open("config/spread_by_hour.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False))
