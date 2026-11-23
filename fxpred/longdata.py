"""Dukascopy から取得した1時間足 (data/duka/*.bi5) を読み込む。
ファイル形式: LZMA 圧縮、1本24バイト (月初からの秒, 始値, 終値, 安値, 高値 を整数(価格x1000), 出来高 float)。ビッグエンディアン。"""
import glob
import lzma
import os
import struct

import pandas as pd

DIR = os.path.join(os.path.dirname(__file__), "..", "data", "duka")


def load(start="2004-01-01") -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(DIR, "USDJPY_*_BID_hour.bi5"))):
        if os.path.getsize(f) == 0:
            continue
        _, y, m, *_ = os.path.basename(f).split("_")
        base = pd.Timestamp(int(y), int(m), 1, tz="UTC")
        b = lzma.decompress(open(f, "rb").read())
        for i in range(len(b) // 24):
            t, o, c, l, h, v = struct.unpack(">5if", b[i * 24:(i + 1) * 24])
            if v > 0:  # 市場が閉まっている時間 (出来高0) は除く
                rows.append((base + pd.Timedelta(seconds=t), o / 1000, h / 1000, l / 1000, c / 1000))
    d = pd.DataFrame(rows, columns=["time", "Open", "High", "Low", "Close"]).set_index("time")
    return d[d.index >= pd.Timestamp(start, tz="UTC")].sort_index()
