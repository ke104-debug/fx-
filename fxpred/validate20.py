"""20年データ (Dukascopy, 2004〜) で作戦と切り替えルールを確かめる。
ルールは 2024-2026 年のデータで決めたので、2004-2023 年は「ルールを決める時に見ていない期間」になる。"""
import pandas as pd

from . import longdata, regime as rg, scan

KEY = {"S": "S 急変の反動(流れの方向)", "H3": "H3 東京レンジのロンドン抜け", "H4": "H4 レンジ逆張り+流れフィルタ", "H5": "H5 指標発表直後の順張り"}


def run():
    d = longdata.load()
    h = rg.prepare(d)
    a = rg.all_trades(h)
    a["期間"] = pd.cut(a.entry.dt.year, [2003, 2008, 2013, 2018, 2023, 2026],
                     labels=["2004-08", "2009-13", "2014-18", "2019-23", "2024-26"])
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(f"USD/JPY 1時間足 {h.index[0]:%Y-%m-%d}〜{h.index[-1]:%Y-%m-%d} ({len(h)}本)")
    print("相場状態の割合:", (h.regime.map(rg.NAMES).value_counts(normalize=True) * 100).round(0).to_dict())
    print("\n■ 作戦 x 相場状態 x 期間 の1回平均R (回数)")
    g = a.groupby(["作戦", "状態", "期間"], observed=True).R.agg(["mean", "size"])
    tbl = g.apply(lambda r: f"{r['mean']:+.2f} ({int(r['size'])})", axis=1).unstack("期間")
    print(tbl.to_string())
    plan = {(KEY[k], s): r for (k, s), r in scan.RULES.items()}
    print("\n■ 切り替え運用 (scan.RULES のリスク設定) を各期間それぞれ100万円から")
    for per, sub in a.groupby("期間", observed=True):
        cv, t = rg.run_switching(sub, plan)
        if len(cv) == 0:
            continue
        cv.index = pd.DatetimeIndex(cv.index)
        mdd = (1 - cv / cv.cummax()).max()
        yrs = (sub.entry.max() - sub.entry.min()).days / 365.25
        print(f"{per}: 100万→{cv.iloc[-1] / 1e4:.0f}万 (年率 {(cv.iloc[-1] / 1e6) ** (1 / yrs) - 1:+.0%}) 最大下落 {mdd:.0%} 取引 {len(t)}回")
    cv, t = rg.run_switching(a[a.entry.dt.year <= 2023], plan)
    cv.index = pd.DatetimeIndex(cv.index)
    y = cv.resample("YE").last().ffill()
    print("\n2004-2023 通し: 年ごとの増減", ((y / y.shift(1).fillna(1e6) - 1) * 100).round(0).astype(int).rename(lambda x: x.year).to_dict())
    print(f"2004-2023 通し: 100万→{cv.iloc[-1] / 1e4:,.0f}万 最大下落 {(1 - cv / cv.cummax()).max():.0%}")
    return h, a


if __name__ == "__main__":
    run()
