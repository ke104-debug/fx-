"""USD/JPY 1時間足 (約2.8年) で売買作戦を検証する。

公正な検証のためのルール:
- シグナルは足の確定後に判断し、次の足の始値で入る (未来の情報を使わない)
- 同じ足で損切りと利確の両方に届いた場合は、損切りが先と見なす (不利な側)
- 往復コスト (スプレッド等) を差し引く。保有は1作戦につき1ポジションまで
- パラメータは事前に決めて固定し、前半/後半に分けて同じ傾向が出るかを見る
"""
import numpy as np
import pandas as pd
import yfinance as yf

PIP = 0.01

# 大きな材料 (米雇用統計・FOMC)。FOMC は公表スケジュールの決定日、雇用統計は原則第1金曜。
FOMC = ["2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07",
        "2024-12-18", "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17",
        "2025-10-29", "2025-12-10", "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
        "2026-09-16", "2026-10-28", "2026-12-09"]


def events(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    ev = [pd.Timestamp(d + " 18:00", tz="UTC") for d in FOMC]
    for m in pd.date_range(index[0].normalize().tz_localize(None), index[-1].tz_localize(None), freq="MS"):
        fri = m + pd.offsets.Week(weekday=4) if m.weekday() != 4 else m
        ev.append(pd.Timestamp(fri.date().isoformat() + " 12:30", tz="UTC"))
    return pd.DatetimeIndex(sorted(ev))


def load() -> pd.DataFrame:
    d = yf.download("USDJPY=X", period="730d", interval="1h", progress=False, auto_adjust=False)
    d.columns = d.columns.get_level_values(0)
    d = d[["Open", "High", "Low", "Close"]].dropna()
    d.index = pd.to_datetime(d.index, utc=True)
    return d


def indicators(d: pd.DataFrame, n: int = 72) -> pd.DataFrame:
    d = d.copy()
    c, h, l = d.Close, d.High, d.Low
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr_w = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * pd.Series(pdm, d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_w
    ndi = 100 * pd.Series(ndm, d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_w
    d["adx"] = (100 * (pdi - ndi).abs() / (pdi + ndi)).ewm(alpha=1 / 14, adjust=False).mean()
    d["sup"] = l.shift(1).rolling(n).min()      # 直前 n 本の安値 = 下値支持線
    d["res"] = h.shift(1).rolling(n).max()      # 直前 n 本の高値 = 上値抵抗線
    d["width"] = d.res - d.sup
    d["drift"] = (c - c.shift(n)).abs()
    d["ema50"], d["ema200"] = c.ewm(span=50).mean(), c.ewm(span=200).mean()
    delta = c.diff()
    g = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    ls = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + g / ls)
    d["mid20"], d["sd20"] = c.rolling(20).mean(), c.rolling(20).std()
    ev = events(d.index)
    pos = ev.searchsorted(d.index)  # 次のイベントまでの時間
    nxt = pd.Series([ev[i] if i < len(ev) else pd.NaT for i in pos], index=d.index)
    d["h_to_event"] = (nxt - d.index.to_series()).dt.total_seconds() / 3600
    return d


# ---------------- 作戦 (各関数は足 i の確定時点で (方向, 損切り価格, 利確価格) か None を返す)

def range_reversal(d, i, both_sides=False, target=0.8, stop_atr=0.5, event_wait=False):
    """ご提案の作戦: レンジ相場で下値付近の反発を買い、上値付近で利確、下値割れで損切り"""
    r = d.iloc[i]
    if not (4 * r.atr <= r.width <= 15 * r.atr) or r.adx >= 25 or r.drift >= 0.5 * r.width:
        return None  # レンジ相場ではない
    if event_wait and not (6 <= r.h_to_event <= 72):
        return None  # 大きな材料待ちの時間帯だけ
    near = 0.15 * r.width
    if r.Low <= r.sup + near and r.Close > r.sup and r.Close > r.Open:       # 下値に触れて陽線で反発
        return 1, r.sup - stop_atr * r.atr, r.sup + target * r.width
    if both_sides and r.High >= r.res - near and r.Close < r.res and r.Close < r.Open:
        return -1, r.res + stop_atr * r.atr, r.res - target * r.width
    return None


def breakout(d, i):
    """順張り: レンジの上抜け/下抜けに付いていく"""
    r, p = d.iloc[i], d.iloc[i - 1]
    if not (4 * r.atr <= r.width <= 15 * r.atr):
        return None
    if r.Close > r.res and p.Close <= p.res:
        risk = max(r.Close - (r.sup + r.res) / 2, 1.0 * r.atr)
        return 1, r.Close - risk, r.Close + 2 * risk
    if r.Close < r.sup and p.Close >= p.sup:
        risk = max((r.sup + r.res) / 2 - r.Close, 1.0 * r.atr)
        return -1, r.Close + risk, r.Close - 2 * risk
    return None


def trend_pullback(d, i):
    """押し目買い/戻り売り: 大きな流れの方向に、一時的な逆行で入る"""
    r = d.iloc[i]
    if r.ema50 > r.ema200 and r.Close > r.ema200 and r.rsi < 35:
        return 1, r.Close - 1.5 * r.atr, r.Close + 3 * r.atr
    if r.ema50 < r.ema200 and r.Close < r.ema200 and r.rsi > 65:
        return -1, r.Close + 1.5 * r.atr, r.Close - 3 * r.atr
    return None


def bollinger_reversion(d, i):
    """ボリンジャー逆張り: 流れの弱い時に ±2σ から中心線への戻りを狙う"""
    r = d.iloc[i]
    if r.adx >= 20:
        return None
    if r.Close < r.mid20 - 2 * r.sd20:
        return 1, r.Close - 1.5 * r.atr, r.mid20
    if r.Close > r.mid20 + 2 * r.sd20:
        return -1, r.Close + 1.5 * r.atr, r.mid20
    return None


# ---------------- 売買シミュレーション

def simulate(d, rule, cost_pips=0.5, max_hold=48, exit_before_event=False, **kw):
    trades, i, n = [], 220, len(d)
    O, H, L, C = d.Open.values, d.High.values, d.Low.values, d.Close.values
    hte = d.h_to_event.values
    while i < n - 2:
        sig = rule(d, i, **kw)
        if sig is None:
            i += 1; continue
        side, stop, tgt = sig
        j = i + 1
        entry = O[j]
        if (side == 1 and not (stop < entry < tgt)) or (side == -1 and not (tgt < entry < stop)):
            i += 1; continue
        exit_px, why = None, ""
        for k in range(j, min(j + max_hold, n)):
            if side == 1:
                if O[k] <= stop: exit_px, why = O[k], "損切り"
                elif L[k] <= stop: exit_px, why = stop, "損切り"
                elif H[k] >= tgt: exit_px, why = (O[k] if O[k] > tgt else tgt), "利確"
            else:
                if O[k] >= stop: exit_px, why = O[k], "損切り"
                elif H[k] >= stop: exit_px, why = stop, "損切り"
                elif L[k] <= tgt: exit_px, why = (O[k] if O[k] < tgt else tgt), "利確"
            if exit_px is None and exit_before_event and hte[k] <= 1:
                exit_px, why = C[k], "材料前に決済"
            if exit_px is not None:
                break
        else:
            k = min(j + max_hold, n) - 1
            exit_px, why = C[k], "時間切れ"
        pips = side * (exit_px - entry) / PIP - cost_pips
        risk = abs(entry - stop) / PIP
        trades.append(dict(entry_time=d.index[j], exit_time=d.index[k], side=side, entry=entry, exit=exit_px,
                           pips=pips, r=pips / risk if risk else 0, why=why))
        i = k + 1
    return pd.DataFrame(trades)


def random_baseline(d, ref: pd.DataFrame, seeds=200, cost_pips=0.5):
    """偶然との比較: 同じ回数・同じ損切り/利確幅で、ランダムな時刻・方向に入った場合の成績分布"""
    if ref.empty:
        return None
    rng = np.random.default_rng(0)
    O, H, L, C = d.Open.values, d.High.values, d.Low.values, d.Close.values
    totals = []
    risk = (ref.entry - ref.entry).abs()  # placeholder for shape
    stops = (ref.r.abs() > 0)
    dist_stop = (ref.pips.abs() / ref.r.abs().replace(0, np.nan)).fillna(20).values  # pips
    dist_tgt = dist_stop * 1.6
    for _ in range(seeds):
        tot = 0.0
        for s_p, t_p in zip(dist_stop, dist_tgt):
            j = rng.integers(220, len(d) - 50)
            side = rng.choice([1, -1])
            e = O[j]; stop = e - side * s_p * PIP; tgt = e + side * t_p * PIP
            px = C[min(j + 47, len(d) - 1)]
            for k in range(j, min(j + 48, len(d))):
                if (side == 1 and L[k] <= stop) or (side == -1 and H[k] >= stop): px = stop; break
                if (side == 1 and H[k] >= tgt) or (side == -1 and L[k] <= tgt): px = tgt; break
            tot += side * (px - e) / PIP - cost_pips
        totals.append(tot)
    return np.array(totals)


def stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return dict(回数=0)
    win, loss = t[t.pips > 0], t[t.pips <= 0]
    eq = t.pips.cumsum()
    pf = win.pips.sum() / -loss.pips.sum() if len(loss) and loss.pips.sum() < 0 else np.inf
    return {"回数": len(t), "勝率": f"{len(win) / len(t):.0%}", "平均利益": round(win.pips.mean(), 1) if len(win) else 0,
            "平均損失": round(loss.pips.mean(), 1) if len(loss) else 0, "PF": round(pf, 2),
            "1回平均": round(t.pips.mean(), 1), "合計pips": round(t.pips.sum()), "最大DD": round((eq.cummax() - eq).max())}
