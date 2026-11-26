"""ユーロドル・ユーロ円の見張り (軽い・厳しめ)。利用者の指示: ドル円が主役。ユーロのペアは資源を割かず、基準を厳しめに。
AI は使わない。Yahoo の1時間足3つ (EUR/USD・EUR/JPY・USD/JPY) だけ。毎時の分析から1回呼ぶ。

3通貨の強さ (ドル・ユーロ・円): 対数で EURUSD = e-u, USDJPY = u-j, EURJPY = e-j、e+u+j=0 とすると
  e = (rEU + rEJ)/3, u = (rUJ - rEU)/3, j = -(rUJ + rEJ)/3  (残り2つの通貨に対してどれだけ動いたか、普段の何倍か)

23年の検証 (提案 P036、1時間足 2004-2026、352通り):
- 厳しめの基準 (作る・確かめともに +0.05R 超、2022年以降プラス、プラスの年60%以上) を満たしたのは12通り。
  偽物の相場 (20日ごとに順番を入れ替え) でも 9〜19通り満たす → 数としては偶然と区別できない
- 一番良かったのは ユーロドル「3つの並び」(24時間で ユーロ/ドルの片方が残り2つに2倍以上強く、もう片方が2倍以上弱い → 強い方へ、
  10時〜翌1時、損切り1.5ATR・利確3ATR): 作る +0.12R / 確かめ +0.53R / 2022年以降 +0.30R、年8回。偽物の最大 (+0.47) をわずかに上回るだけ
→ 通知はしない。影の運用で記録し (journal/eur_shadow.csv)、夜の検証部屋で答え合わせ。朝の報告に3通貨の強さを1行"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
LOG = os.path.join(ROOT, "journal", "eur_shadow.csv")
RULES = {   # 名前: (ペア, 種類, 倍率)
    "ユーロドル 3つの並び24時間": ("EURUSD", "並び", 2.0),
    "ユーロドル 急変に付いていく": ("EURUSD", "急変", 4.0),
}


def fetch(period="1y") -> pd.DataFrame | None:
    try:
        import yfinance as yf
        d = yf.download(["EURUSD=X", "EURJPY=X", "JPY=X"], period=period, interval="1h", progress=False, auto_adjust=False)
        out = {}
        for k, t in (("EURUSD", "EURUSD=X"), ("EURJPY", "EURJPY=X"), ("USDJPY", "JPY=X")):
            out[k] = d.xs(t, axis=1, level=1)[["Open", "High", "Low", "Close"]]
        x = pd.concat(out, axis=1).dropna()
        x.index = pd.to_datetime(x.index, utc=True)
        return x
    except Exception as e:
        print("ユーロのデータを取れませんでした:", e)
        return None


def strengths(x: pd.DataFrame, n: int) -> pd.DataFrame:
    rEU, rEJ, rUJ = (np.log(x[p].Close).diff(n) for p in ("EURUSD", "EURJPY", "USDJPY"))
    s = pd.DataFrame({"ユーロ": (rEU + rEJ) / 3, "ドル": (rUJ - rEU) / 3, "円": -(rUJ + rEJ) / 3})
    return s / s.rolling(2000, min_periods=500).std()


def _atr(p: pd.DataFrame) -> pd.Series:
    c = p.Close
    tr = pd.concat([p.High - p.Low, (p.High - c.shift()).abs(), (p.Low - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def signals(x: pd.DataFrame) -> pd.DataFrame:
    """各足の合図 (+1 買い / -1 売り / 0)。日本時間 10時〜翌1時だけ"""
    s24 = strengths(x, 24)
    e, u, j = s24["ユーロ"], s24["ドル"], s24["円"]
    jh = (x.index.hour + 9) % 24
    act = (jh >= 10) | (jh < 1)
    out = pd.DataFrame(index=x.index)
    k = RULES["ユーロドル 3つの並び24時間"][2]
    up = (e >= k) & (u <= -k) & (e > j) & (j > u)
    dn = (e <= -k) & (u >= k) & (e < j) & (j < u)
    out["ユーロドル 3つの並び24時間"] = np.where(act & up, 1, np.where(act & dn, -1, 0))
    r3 = np.log(x["EURUSD"].Close).diff(3)
    z3 = r3 / r3.rolling(2000, min_periods=500).std()
    k = RULES["ユーロドル 急変に付いていく"][2]
    out["ユーロドル 急変に付いていく"] = np.where(act & (z3 >= k), 1, np.where(act & (z3 <= -k), -1, 0))
    return out


def run(dry=False) -> list[dict]:
    """毎時: 確定した最新の足に合図があれば記録 (同じ作戦は6時間に1回)"""
    x = fetch()
    if x is None or len(x) < 600:
        return []
    x = x.iloc[:-1]                              # まだ確定していない最新の足は使わない
    sg = signals(x)
    atr = _atr(x["EURUSD"])
    t = x.index[-1]
    old = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=["time_utc", "rule"])
    rows = []
    for rule in sg.columns:
        s = int(sg[rule].iat[-1])
        if not s:
            continue
        recent = old[(old.rule == rule) & (pd.to_datetime(old.time_utc, utc=True) > t - pd.Timedelta(hours=6))]
        if len(recent):
            continue
        c, a = float(x["EURUSD"].Close.iat[-1]), float(atr.iat[-1])
        rows.append({"time_utc": f"{t:%Y-%m-%d %H:%M}", "rule": rule, "side": "買い" if s == 1 else "売り", "price": round(c, 5),
                     "stop": round(c - s * 1.5 * a, 5), "target": round(c + s * 3.0 * a, 5), "result_r": ""})
    if rows and not dry:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        pd.concat([old, pd.DataFrame(rows)]).to_csv(LOG, index=False)
    return rows


def score() -> list[str]:
    """夜の検証部屋: 答えの出た記録を採点 (損切り・利確・48時間)"""
    if not os.path.exists(LOG):
        return ["ユーロドルの影の運用: まだ合図なし (年に十数回の見込み)"]
    d = pd.read_csv(LOG)
    x = fetch("60d")
    if x is not None:
        p = x["EURUSD"]
        for i, r in d[d.result_r.isna() | (d.result_r.astype(str) == "")].iterrows():
            t = pd.Timestamp(r.time_utc, tz="UTC")
            after = p[p.index > t]
            if after.empty:
                continue
            s = 1 if r.side == "買い" else -1
            e = after.Open.iat[0]
            risk = abs(r.price - r.stop)
            res = None
            for k, (h, lo) in enumerate(zip(after.High, after.Low)):
                if (s == 1 and lo <= r.stop) or (s == -1 and h >= r.stop):
                    res = (s * (r.stop - e) - 0.00005) / risk; break
                if (s == 1 and h >= r.target) or (s == -1 and lo <= r.target):
                    res = (s * (r.target - e) - 0.00005) / risk; break
                if k >= 47:
                    res = (s * (after.Close.iat[k] - e) - 0.00005) / risk; break
            if res is not None:
                d.loc[i, "result_r"] = round(res, 3)
        d.to_csv(LOG, index=False)
    done = pd.to_numeric(d.result_r, errors="coerce").dropna()
    return [f"ユーロドルの影の運用 (通知なし): 合図 {len(d)}回、答えが出た {len(done)}回 平均 {done.mean() if len(done) else 0:+.2f}R"
            + " (23年の検証: 3つの並び 確かめ +0.53R・年8回。偶然と区別できないので記録で確かめる)"]


def line(x: pd.DataFrame | None = None) -> str:
    """朝の報告用: 3通貨の強さ (24時間) とユーロドル・ユーロ円の見方"""
    x = x if x is not None else fetch("1y")
    if x is None or len(x) < 600:
        return "3通貨の強さ: 取得できず"
    s = strengths(x, 24).iloc[-1]
    order = s.sort_values(ascending=False)
    txt = "3通貨の強さ (24時間): " + " > ".join(f"{k} {v:+.1f}" for k, v in order.items())
    gap = s["ユーロ"] - s["ドル"]
    if abs(gap) >= 2:
        txt += f"\nユーロドル: {'ユーロ高・ドル安' if gap > 0 else 'ドル高・ユーロ安'}がはっきり (差 {gap:+.1f})。強い方へ動きやすい形だが、合図は記録のみ"
    else:
        txt += "\nユーロドル・ユーロ円: 目立った偏りなし (見送り)"
    return txt
