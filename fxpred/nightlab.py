"""夜の検証部屋 (利用者が寝ている時間に、シミュレーションと記録の答え合わせを多めに回す)。
GitHub Actions の fx-lab (日本時間 3時台、火〜土) から呼ぶ。公開リポジトリなので Actions は無料。AI は使わない。

1. 15分足の予想 (影の運用・提案 P026): 前の晩に作ったモデルで、直近の15分足 (Yahoo) を予想し直して記録する。
   モデルは「その晩までの1分足」だけで作り、記録するのはモデルを作った後の足だけ (未来の情報は混ざらない)。
   確率60%以上 (または40%以下) の足を記録し (成績は65%以上と分けて見る)、+1ATR と -1ATR のどちらに先に届いたかで答え合わせ (往復0.8pips を引く)。
   通知はしない (ルールは固定中)。
2. モデルの作り直し: 毎週土曜の朝 (または無い時)。2010年以降の15分足 (data-m1 ブランチの1分足から作る)。
3. まとめ: reports/nightlab.md (記録の合図の数・的中・平均R)。週次の報告で使う。"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL = os.path.join(ROOT, "models", "m15.txt")
META = os.path.join(ROOT, "models", "m15.json")
LOG = os.path.join(ROOT, "journal", "m15_shadow.csv")
REPORT = os.path.join(ROOT, "reports", "nightlab.md")
TH, HOLD, COST = 0.60, 24, 0.8   # 記録は60%以上 (成績は65%以上と分けて見る)


def _label(d):
    from .mlpredict import labels
    return labels(d, 1.0, HOLD)


def train():
    import lightgbm as lgb
    from . import explore as ex, m1data, mlpredict as ml
    m1 = m1data.load()
    d = ex.bars(tf="15min", src=m1[m1.index >= "2009-01-01"])
    d = d[d.atr > 0]
    X, y = ml.features(d), _label(d)
    ok = np.flatnonzero((d.index.year >= 2010) & ~np.isnan(y))[:-30][::4]
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(X.iloc[ok], y[ok])
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    m.booster_.save_model(MODEL)
    json.dump({"made_utc": f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M}", "data_until": f"{d.index[-1]:%Y-%m-%d %H:%M}",
               "columns": list(X.columns), "n": int(len(ok))}, open(META, "w"), ensure_ascii=False)
    print(f"15分足のモデルを作り直し: 学習 {len(ok)}本 (1分足は {d.index[-1]:%Y-%m-%d} まで)")


def _yahoo15():
    import yfinance as yf
    x = yf.download("USDJPY=X", period="60d", interval="15m", progress=False, auto_adjust=False)
    if x is None or len(x) == 0:
        return None
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x.index = pd.to_datetime(x.index, utc=True)
    return x[["Open", "High", "Low", "Close"]].dropna()


def shadow():
    """モデルを作った後の15分足を予想し、答えが出たものを記録 (同じ足は1回だけ)"""
    import lightgbm as lgb
    from . import explore as ex, mlpredict as ml
    if not os.path.exists(MODEL):
        return 0
    meta = json.load(open(META))
    made = pd.Timestamp(meta["made_utc"], tz="UTC")
    raw = _yahoo15()
    if raw is None or len(raw) < 1500:
        print("Yahoo の15分足を取れませんでした")
        return 0
    d = ex.bars(tf=None, src=raw.copy())
    d = d[d.atr > 0]
    X = ml.features(d)[meta["columns"]]
    p = lgb.Booster(model_file=MODEL).predict(X)
    y = _label(d)
    jh = d.jst_h.values
    sel = (d.index >= made) & ((p >= TH) | (p <= 1 - TH)) & ((jh >= 10) | (jh < 1)) & ~np.isnan(y)
    rows = []
    for i in np.flatnonzero(sel):
        side = 1 if p[i] > .5 else -1
        win = (side == 1) == (y[i] == 1)
        r = (1.0 if win else -1.0) - COST / 100 / d.atr.values[i]
        rows.append({"time_utc": f"{d.index[i]:%Y-%m-%d %H:%M}", "side": "買い" if side == 1 else "売り", "prob": round(float(p[i]), 3),
                     "entry": round(float(d.Open.values[i + 1]), 3), "atr_pips": round(float(d.atr.values[i]) * 100, 1),
                     "win": int(win), "r": round(r, 3), "model": meta["made_utc"]})
    if not rows:
        return 0
    new = pd.DataFrame(rows)
    old = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=new.columns)
    new = new[~new.time_utc.isin(set(old.time_utc))]
    if len(new):
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        pd.concat([old, new]).to_csv(LOG, index=False)
    return len(new)


def summary() -> list[str]:
    if not os.path.exists(LOG):
        return ["15分足の予想 (影の運用): まだ記録なし"]
    d = pd.read_csv(LOG)
    t = pd.to_datetime(d.time_utc, utc=True)
    out = ["15分足の予想 (影の運用・通知なし)"]
    strong = (d.prob - 0.5).abs() >= 0.15
    for name, x in (("確率65%以上", d[strong]), ("確率60〜65%", d[~strong])):
        if len(x):
            out.append(f"  {name}: {len(x)}回 的中{x.win.mean():.0%} 平均{x.r.mean():+.2f}R 合計{x.r.sum():+.1f}R")
    w = d[(t >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)) & strong]
    if len(w):
        out.append(f"  直近7日 (65%以上): {len(w)}回 的中{w.win.mean():.0%} 平均{w.r.mean():+.2f}R")
    out.append("  (23年の検証 確かめる期間: 65%以上 +0.14R・的中61% / 60%以上 +0.04R。作る期間はほぼ0。数百回たまるまで判断しない)")
    return out


def run(force_train=False):
    now = pd.Timestamp.now(tz="Asia/Tokyo")
    n = shadow()      # 先に古いモデルで答え合わせ (作り直す前のモデルの成績として残す)
    print(f"15分足の予想の記録: {n}回 追加")
    if force_train or not os.path.exists(MODEL) or now.weekday() == 5:
        try:
            train()
        except Exception as e:
            print("モデルを作り直せませんでした:", e)
    try:   # 点検部隊 (毎晩の詳しい点検: ログ・5分監視の途切れ・AI のコード点検)
        from . import inspector, scan, regime as rg, notify
        for kind, title, body in inspector.run(rg.prepare(scan.load_recent()), deep=True):
            notify.send(title, body, color=16776960)
    except Exception as e:
        print("点検部隊のエラー:", e)
    lines = summary()
    try:
        from .org import eurwatch
        lines += eurwatch.score()
    except Exception as e:
        print("ユーロの答え合わせのエラー:", e)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write(f"# 夜の検証部屋 ({now:%Y-%m-%d %H:%M} 日本時間)\n\n" + "\n".join(f"- {x}" for x in lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    run(force_train="--train" in sys.argv)
