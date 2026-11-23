"""今のシステム (毎時の分析・5分監視・急変通知) を過去に動かしたら、どんな通知を出し、結果がどうだったかを再現する。
各時点ではその時点までのデータだけを使う。ニュースの見出しは過去分が取れないため、急変担当の原因推定は含めない。"""
import numpy as np
import pandas as pd
import yfinance as yf

from . import diagnose, edge, features as ft, model, regime as rg, scan, specialists as sp, strategies as st

HOLD = {"H3": 14, "H4": 48, "H5": 24, "S": 72}


def _vix():
    v = yf.download("^VIX", period="730d", interval="1h", progress=False)["Close"].squeeze().dropna()
    v.index = pd.to_datetime(v.index, utc=True)
    return v


def _outcome(h, j, side, entry, stop, target, hold):
    w = h.iloc[j:j + hold]
    for b in w.itertuples():
        if (side == 1 and b.Low <= stop) or (side == -1 and b.High >= stop):
            return "損切り", side * (stop - entry) * 100 - 0.5
        if (side == 1 and b.High >= target) or (side == -1 and b.Low <= target):
            return "利確", side * (target - entry) * 100 - 0.5
    return "期限切れ", side * (w.Close.iat[-1] - entry) * 100 - 0.5


def hourly(h: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """毎時の分析 (scan.run と同じ判断) を全期間で再現"""
    raw = h[["Open", "High", "Low", "Close"]]
    rows, models = [], {}
    events = st.events(h.index)
    for i in range(700, len(h) - 2):
        t = h.index[i]
        reg = rg.NAMES[h.regime.iat[i]]
        sigs = []
        if ("S", reg) in scan.RULES:
            s = scan.shock_signal(h.iloc[:i + 1])
            if s: sigs.append(("S", s))
        for key, fn in scan.RULE_FN.items():
            if (key, reg) in scan.RULES:
                r = fn(h, i)
                if r:
                    side, stop, tgt, trail = r
                    sigs.append((key, dict(side=side, entry=h.Close.iat[i], stop=stop, target=tgt)))
        for key, s in sigs:
            sub = h.iloc[:i + 1]
            tech = diagnose.technical(raw.iloc[:i + 1])
            tf = {k: tech[k]["trend"] for k in ("15分足", "1時間足", "4時間足")}
            month = t.strftime("%Y-%m")
            if month not in models:   # AIモデルは毎月、その時点までのデータで学習し直す
                f = ft.build(raw.loc[:t.replace(day=1, hour=0)], "USDJPY=X", np.log(raw.Close).diff(), horizon=4)
                models[month] = model.train(f) if len(f) > 500 else None
            p_up = 0.5
            if models[month] is not None:
                fx = ft.build(raw.iloc[:i + 1], "USDJPY=X", np.log(raw.Close).diff(), horizon=4).tail(1)
                p_up = float(model.predict_proba(models[month], fx).iloc[0])
            vv = vix.loc[:t]
            vix_chg = float(vv.iat[-1] - vv.iat[-5]) if len(vv) > 5 else None
            anom = edge.anomaly(sub)
            n_ok, ok, ng = edge.confluence(s["side"], tf, scan.near_level(sub, s["side"]), p_up, vix_chg, reg, True)
            stop_why = edge.no_trade(sub, t + pd.Timedelta(hours=1), reg, tf, anom) if key != "S" else []
            serious = [w for w in stop_why if w.startswith(("重要指標", "急変の最中", "週末"))]
            views = {"急変担当": sp.shock_view(sub, anom, []), "レンジ担当": sp.range_view(sub, sub.rsi.iat[-1]),
                     "トレンド担当": sp.trend_view(sub)}
            coord = sp.coordinate(reg, views, {})
            agree = 1 if coord["verdict"] == ("買い寄り" if s["side"] == 1 else "売り寄り") else \
                -1 if coord["verdict"] == ("売り寄り" if s["side"] == 1 else "買い寄り") else 0
            score = edge.confidence(ok, agree, len(stop_why))
            awake = scan.awake(t + pd.Timedelta(hours=1), events)
            sent = (key == "S") or (score >= edge.SEND_AT and awake)
            tgt = s["target"] if s.get("target") else s["entry"] + s["side"] * 2 * abs(s["entry"] - s["stop"])
            res, pips = _outcome(h, i + 1, s["side"], h.Open.iat[i + 1], s["stop"], tgt, HOLD[key])
            rows.append(dict(time=t, key=key, regime=reg, side=s["side"], score=score, agree=agree, awake=awake,
                             conds="|".join(x.split(" ")[0] for x in ok), n_warn=len(stop_why),
                             serious=bool(serious), sent=sent and not serious, result=res, pips=round(pips, 1),
                             stop=s["stop"], target=tgt, hold=HOLD[key]))
    return pd.DataFrame(rows)


def quick5(h: pd.DataFrame) -> pd.DataFrame:
    """5分監視 (下値・上値に接近 → 5分足で反発確認 → 信頼度) を直近の5分足で再現"""
    m5 = yf.download("USDJPY=X", period="60d", interval="5m", progress=False, auto_adjust=False)
    m5.columns = m5.columns.get_level_values(0)
    m5 = m5[["Open", "High", "Low", "Close"]].dropna(); m5.index = pd.to_datetime(m5.index, utc=True)
    lv = {}
    for t in h.index[h.index >= m5.index[0] - pd.Timedelta(hours=1)]:
        sub = h.loc[:t]
        rv = sp.range_view(sub, sub.rsi.iat[-1]); lv[t + pd.Timedelta(hours=1)] = (rv["support"], rv["resistance"], rg.NAMES[sub.regime.iat[-1]])
    L = pd.DataFrame(lv, index=["sup", "res", "reg"]).T.sort_index()
    m5 = m5.join(L.reindex(m5.index, method="ffill")).dropna()
    O, H, Lo, C = (m5[c].values.astype(float) for c in ["Open", "High", "Low", "Close"])
    rows, busy = [], -1
    events = st.events(h.index)
    for i in range(3, len(m5) - 1):
        if i <= busy: continue
        for side, lvl, other in ((1, m5.sup.iat[i], m5.res.iat[i]), (-1, m5.res.iat[i], m5.sup.iat[i])):
            if abs(C[i] - lvl) > 0.10: continue
            touched = (Lo[i - 2:i + 1].min() <= lvl + 0.05) if side == 1 else (H[i - 2:i + 1].max() >= lvl - 0.05)
            turned = (C[i] > O[i] and C[i] > C[i - 1]) if side == 1 else (C[i] < O[i] and C[i] < C[i - 1])
            if not (touched and turned): continue
            e, stop, tgt = O[i + 1], lvl - side * 0.20, other - side * 0.10
            if side * (tgt - e) <= 0.05 or side * (e - stop) <= 0.05: continue
            t = m5.index[i]
            sub = h.loc[:t - pd.Timedelta(minutes=1)]
            tech = diagnose.technical(sub[["Open", "High", "Low", "Close"]])
            tf = {k: tech[k]["trend"] for k in ("15分足", "1時間足", "4時間足")}
            reg = m5.reg.iat[i]
            n_ok, ok, ng = edge.confluence(side, tf, "下値・上値で反発確認", 0.5, None, reg, False)
            anom = edge.anomaly(sub)
            views = {"急変担当": sp.shock_view(sub, anom, []), "レンジ担当": sp.range_view(sub, sub.rsi.iat[-1]),
                     "トレンド担当": sp.trend_view(sub)}
            coord = sp.coordinate(reg, views, {})
            agree = 1 if coord["verdict"] == ("買い寄り" if side == 1 else "売り寄り") else \
                -1 if coord["verdict"] == ("売り寄り" if side == 1 else "買い寄り") else 0
            stop_why = edge.no_trade(sub, t, reg, tf, anom)
            score = edge.confidence(ok, agree, len(stop_why))
            sent = score >= edge.SEND_AT_QUICK and scan.awake(t, events)
            res, pips = None, None
            for k in range(i + 1, min(i + 577, len(m5))):
                if (side == 1 and Lo[k] <= stop) or (side == -1 and H[k] >= stop): res, pips = "損切り", side * (stop - e) * 100 - 0.5; break
                if (side == 1 and H[k] >= tgt) or (side == -1 and Lo[k] <= tgt): res, pips = "利確", side * (tgt - e) * 100 - 0.5; break
            if res is None: res, pips = "期限切れ", side * (C[k] - e) * 100 - 0.5
            rows.append(dict(time=t, side=side, score=score, agree=agree, sent=sent, result=res, pips=round(pips, 1),
                             stop=stop, target=tgt, hold=48, ready=m5.index[i + 1]))
            busy = k
            break
    return pd.DataFrame(rows)


def shocks(h: pd.DataFrame) -> pd.DataFrame:
    """急変の通知 (毎時の判定) のあと、相場がどう動いたか"""
    rows, last = [], None
    for i in range(500, len(h) - 25):
        a = edge.anomaly(h.iloc[:i + 1])
        if a.get("level") != "急変": continue
        day = h.index[i].strftime("%Y-%m-%d")
        if last == day: continue
        last = day
        c = h.Close.iat[i]; d = 1 if a["move"] > 0 else -1
        f4, f24 = h.Close.iat[i + 4] - c, h.Close.iat[i + 24] - c
        rows.append(dict(time=h.index[i], dir="急騰" if d > 0 else "急落", move=a["move"], next4h=round(f4, 2), next24h=round(f24, 2),
                         cont24=(f24 * d > 0)))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str) -> list[str]:
    out = [f"■ {label}"]
    if df.empty:
        return out + ["  該当なし"]
    s, k = df[df.sent], df[~df.sent]
    def line(x, name):
        if x.empty: return f"  {name}: 0回"
        return (f"  {name}: {len(x)}回 勝率{(x.pips > 0).mean():.0%} 合計{x.pips.sum():+.0f}pips 1回平均{x.pips.mean():+.1f} "
                f"(利確{(x.result == '利確').sum()} 損切り{(x.result == '損切り').sum()} 期限切れ{(x.result == '期限切れ').sum()})")
    out += [line(s, "通知した"), line(k, "見送った"), line(df, "(参考) 全部入った場合")]
    if len(s):
        b = pd.cut(s.score, [0, 64, 74, 100], labels=["55〜64%", "65〜74%", "75%以上"])
        for lab, g in s.groupby(b, observed=True):
            out.append("   " + line(g, f"信頼度{lab}").strip())
        span = max((df.time.max() - df.time.min()).days / 30.4, 1)
        out.append(f"  通知の頻度: 月{len(s) / span:.1f}回")
    return out


def delayed(m5: pd.DataFrame, sig: pd.DataFrame, delay_min: int, manual_exit: bool) -> pd.DataFrame:
    """通知が出た時刻 (ready) から delay_min 分後の価格で入る。損切り・利確の価格は通知どおり。
    入る時点で損切り・利確をすでに越えていたら「見送り (機会を逃した)」。
    manual_exit=True: 決済も通知を見てから delay_min 分後の価格で行う (注文を入れておかない場合)"""
    idx, O, H, L, C = m5.index, m5.Open.values, m5.High.values, m5.Low.values, m5.Close.values
    rows = []
    for r in sig.itertuples():
        t_in = r.ready + pd.Timedelta(minutes=delay_min)
        j = idx.searchsorted(t_in)
        if j >= len(idx) - 1:
            continue
        e = O[j]
        side, stop, tgt = r.side, r.stop, r.target
        if side * (e - stop) <= 0 or side * (tgt - e) <= 0:
            rows.append(dict(time=r.time, result="見送り", pips=0.0)); continue
        end = min(j + r.hold * 12, len(idx) - 1)
        res, px = "期限切れ", C[end]
        for k in range(j, end + 1):
            hit_stop = (side == 1 and L[k] <= stop) or (side == -1 and H[k] >= stop)
            hit_tgt = (side == 1 and H[k] >= tgt) or (side == -1 and L[k] <= tgt)
            if hit_stop or hit_tgt:
                res = "損切り" if hit_stop else "利確"
                if manual_exit:   # 通知から delay 分後の価格で決済
                    k2 = min(k + max(delay_min // 5, 1), len(idx) - 1)
                    px = O[k2]
                else:
                    px = stop if hit_stop else tgt
                break
        rows.append(dict(time=r.time, result=res, pips=round(side * (px - e) * 100 - 0.5, 1)))
    return pd.DataFrame(rows)
