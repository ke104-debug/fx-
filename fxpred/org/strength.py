"""ドル主導か円主導かを見分ける担当 (相場分析部隊)。EUR/USD と EUR/JPY の1時間足から、
ドルの強さ = EUR/USD の逆の動き、円の強さ = EUR/JPY の逆の動き を、普段の動きの何倍か (z) で表す。

23年の検証 (reports/strength.md):
- ドル円が大きく上がった時、ドル主導 (ドルが強く、円は強くない) の上げは24時間後も続きやすい (作る +3〜+16pips / 確かめ +8〜+13pips)。円安主導の上げは続かない
- 急変に付いていく取引で、ドルか円の強さのどちらかが取引と逆向きの時は成績が悪い (作る -0.16〜+0.03R / 確かめ -0.10〜-0.26R、全体は +0.24〜+0.57R)
→ 急変の合図には「強さが逆」の注意を付け、影の運用で記録する (止めるかは記録を見てから)"""
import numpy as np
import pandas as pd


def fetch(period="60d") -> pd.DataFrame | None:
    """Yahoo の1時間足 (EUR/USD・EUR/JPY)。為替の1時間足は日足と違って時刻のずれはない"""
    try:
        import yfinance as yf
        d = yf.download(["EURUSD=X", "EURJPY=X"], period=period, interval="1h", progress=False, auto_adjust=False)["Close"]
        d.index = pd.to_datetime(d.index, utc=True)
        return d.rename(columns={"EURUSD=X": "eu", "EURJPY=X": "ej"}).dropna()
    except Exception:
        return None


def zscores(eu: pd.Series, ej: pd.Series, n: int) -> pd.DataFrame:
    lr_u, lr_j = -np.log(eu).diff(n), -np.log(ej).diff(n)
    sd_u = np.log(eu).diff(n).rolling(500, min_periods=100).std()
    sd_j = np.log(ej).diff(n).rolling(500, min_periods=100).std()
    return pd.DataFrame({"usd": lr_u / sd_u, "jpy": lr_j / sd_j})


def now(uj_close: pd.Series) -> dict | None:
    x = fetch()
    if x is None or len(x) < 150:
        return None
    out = {}
    for n in (4, 24):
        z = zscores(x.eu, x.ej, n).iloc[-1]
        out[f"usd{n}"], out[f"jpy{n}"] = float(z.usd), float(z.jpy)
    u, j = out["usd24"], out["jpy24"]
    lead = ("ドル高" if u > 0.5 else "ドル安" if u < -0.5 else "") + ("・" if abs(u) > 0.5 and abs(j) > 0.5 else "") + \
           ("円高" if j > 0.5 else "円安" if j < -0.5 else "")
    out["text"] = f"24時間: ドル {u:+.1f} / 円 {j:+.1f} (普段の動きの倍率)" + (f" → {lead}が主導" if lead else " → どちらも目立たない")
    # ドル主導の上げ: 直近4時間でドル円が大きく上がり、ドルが強く円は強くない
    uj = np.log(uj_close)
    m4 = uj.diff(4)
    z4 = (m4 / m4.rolling(500, min_periods=100).std()).iat[-1]
    out["usd_led_up"] = bool(z4 >= 2 and out["usd4"] > 1 and out["jpy4"] < 1)
    out["bias"] = 1.0 if out["usd_led_up"] else 0.0
    return out


def against(side: int, st: dict | None) -> bool:
    """取引の向きに対して、ドルか円の強さのどちらかが逆向き (24時間で0.5倍以上)"""
    if not st:
        return False
    return side * st["usd24"] < -0.5 or side * (-st["jpy24"]) < -0.5


def word(z: float) -> str:
    return "強い" if z >= 1 else "やや強い" if z >= 0.5 else "弱い" if z <= -1 else "やや弱い" if z <= -0.5 else "ふつう"


def fetch5(period="30d") -> pd.DataFrame | None:
    """Yahoo の5分足 (EUR/USD・EUR/JPY・USD/JPY)。直近30日"""
    try:
        import yfinance as yf
        d = yf.download(["EURUSD=X", "EURJPY=X", "USDJPY=X"], period=period, interval="5m", progress=False, auto_adjust=False)["Close"]
        d.index = pd.to_datetime(d.index, utc=True)
        return d.rename(columns={"EURUSD=X": "eu", "EURJPY=X": "ej", "USDJPY=X": "uj"}).dropna()
    except Exception:
        return None


def line(uj_close=None) -> str:
    """レポート用: ドルの強さ・円の強さ (1時間・4時間・24時間、普段の何倍か) と、ドル円の動きのうちドル要因・円要因の pips。
    ドル円 = ユーロ円 ÷ ユーロドル なので、ドル円の動き (対数) = ドルの強さ − 円の強さ にぴったり分けられる。
    普段の大きさは、直近30日の同じ長さの動きのばらつき。5分足が取れない時は1時間足 (10日) で"""
    x = fetch5()
    if x is not None and len(x) > 2000:
        per = {"1時間": 12, "4時間": 48, "24時間": 288}
    else:
        x = fetch("10d")
        if x is None or len(x) < 30:
            return "ドル・円の強さ: 取得できず"
        per = {"4時間": 4, "24時間": 24}
    out = {}
    for k, n in per.items():
        lr_u, lr_j = -np.log(x.eu).diff(n), -np.log(x.ej).diff(n)
        out[k] = (lr_u.iat[-1], lr_j.iat[-1], lr_u.iat[-1] / (lr_u.std() or 1e-9), lr_j.iat[-1] / (lr_j.std() or 1e-9))
    px = float(x.uj.iat[-1]) if "uj" in x else 155.0
    fmt = lambda i: " / ".join(f"{k} {v[i]:+.1f} ({word(v[i])})" for k, v in out.items())
    lines = [f"ドルの強さ: {fmt(2)}", f"円の強さ: {fmt(3)}"]
    k4 = "4時間"
    lu, lj, zu, zj = out[k4]
    du, dj = lu * px * 100, -lj * px * 100          # ドル円の pips のうち、ドルが動かした分・円が動かした分
    lines.append(f"ドル円の直近4時間 {du + dj:+.0f}pips = ドル要因 {du:+.0f} + 円要因 {dj:+.0f}")
    if abs(zu) >= 0.5 or abs(zj) >= 0.5:
        lead = ("ドル" + ("高" if zu > 0 else "安")) if abs(zu) >= abs(zj) else ("円" + ("高" if zj > 0 else "安"))
        lines.append(f"→ 直近4時間は{lead}が主導")
    else:
        lines.append("→ 直近4時間はどちらも目立たない")
    return "\n".join(lines)
