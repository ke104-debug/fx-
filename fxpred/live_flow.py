"""通知する作戦「流れ買い・長期」(利用者の承認 2026-11-27)。研究の関数 quest.fam_trend をそのまま使う (検証と通知の食い違いを防ぐ)。
中身: 日本時間10時の日足で、20日平均 > 100日平均 (上昇の流れ) の間は買いを持ち続ける (買いのみ・スワップを受け取る)。
  損切りは入った値から日足ATR×2、その後は最も伸びた値から日足ATR×5 で追いかける。流れが下向きに変わったら (10時の判断で) 決済。
  決済の後も流れが上向きなら、次の日の10時に入り直す。
23年 (1時間足・往復0.8pips・スワップ込み): 106回 (年4.6回)・+0.40R (t=2.0)・作る +0.27R / 確かめ +0.57R / 2022年以降 +0.62R・勝率38%・最大の落ち込み -12.7R。
速さを変えた3通り (EMA20/100・10/50、追いかけ3ATR/5ATR) すべてで両期間プラス。
毎時の分析から呼ぶ。研究の関数で「今持っているはずの取引」を求め、見張りの記録 (positions.json, key=TF) と合わせる。"""
import pandas as pd

KEY = "TF"
FAST, SLOW, TRAIL_ATR, SL_ATR = 20, 100, 5.0, 2.0


def _hourly_2y() -> pd.DataFrame:
    import yfinance as yf
    d = yf.download("USDJPY=X", period="730d", interval="1h", progress=False, auto_adjust=False)
    d.columns = d.columns.get_level_values(0)
    d = d[["Open", "High", "Low", "Close"]].dropna()
    d.index = pd.to_datetime(d.index, utc=True)
    return d[d.index + pd.Timedelta(hours=1) <= pd.Timestamp.now(tz="UTC")]     # 確定した足だけ


def state(h: pd.DataFrame | None = None) -> dict | None:
    """研究の関数で、今持っているはずの取引 (なければ None) と、今の損切り"""
    from . import quest as q
    h = h if h is not None else _hourly_2y()
    dd = q.daily(h)
    t = q.fam_trend(h, dd, kind="ema", fast=FAST, slow=SLOW, trail=TRAIL_ATR, sl=SL_ATR, only=1)
    if t.empty:
        return None
    last = t.iloc[-1]
    if last.exit_time != h.index[-1]:      # 最後の取引はもう終わっている
        return {"open": False, "last": last, "h": h}
    i = h.index.get_loc(last.time)
    e = float(h.Open.iat[i + 1])
    j = dd.index.get_loc(last.time)
    a = float(dd.atr.iat[j])
    best = float(h.High.iloc[i + 1:].max())
    stop = max(e - SL_ATR * a, best - TRAIL_ATR * a)
    return {"open": True, "entry_time": h.index[i + 1], "entry": e, "atr": a, "best": best, "stop": stop, "h": h, "last": last}


def check(st: dict, dry=False) -> list[tuple[str, str, str]]:
    """毎時: 見張りの記録と研究の関数の答えを合わせ、新しい買い・損切りの引き上げ・決済を知らせる"""
    from . import positions, fmt, scan
    s = state()
    if s is None:
        return []
    h = s["h"]
    px = float(h.Close.iat[-1])
    ps = positions.load()
    mine = [p for p in ps if p.get("key") == KEY]
    out = []
    if s["open"]:
        if not mine:
            fresh = pd.Timestamp.now(tz="UTC") - s["entry_time"] <= pd.Timedelta(hours=26)
            _sa = __import__("fxpred.signal_audit", fromlist=["x"])
            blk, notes = _sa.gate(1)     # (止める?, 注意)
            notes = [_sa.cost_gate(px, s["stop"])[1]] + notes
            if blk:
                print("(流れ買い) 最終関門で止めた:", notes)
                return []
            reasons = [f"日足の流れが上向き (20日平均 > 100日平均)", "23年の検証で両方の期間がプラス (+0.40R・年4〜5回)"]
            cautions = ["数日〜数週間持つ取引。損切りは広め (日足の値幅の2倍) なので量を小さく",
                        f"利確は置かずに伸ばす。最も伸びた値から {TRAIL_ATR:.0f}×日足ATR ({TRAIL_ATR * s['atr']:.2f}円) 後ろに損切りを付いていかせる",
                        "流れが下向きに変わったら (毎朝10時の判断) 決済を知らせる"] + notes
            if fresh:
                title, body = fmt.signal(1, px, s["stop"], None, 60, reasons, cautions, "本日中 (遅れても成績はほぼ変わらない)", None)
            else:
                title = f"📈 流れ買い (保有中の目安) USD/JPY"
                body = (f"この作戦は {(s['entry_time'] + pd.Timedelta(hours=9)):%m/%d %H:%M} に **{s['entry']:.3f}** で買いに入り、今も買いを持っている状態です\n"
                        f"今 {px:.3f} ({(px - s['entry']) * 100:+.0f}pips) / 今の損切りの目安 **{s['stop']:.3f}**\n"
                        "・これから入るなら、損切りは上の値。量は損切りまでの幅に合わせて小さく\n"
                        + "\n".join("・" + c for c in cautions))
            if not dry:
                scan.log_signal(pd.Timestamp.now(tz="UTC").floor("min"), KEY, {"side": 1, "entry": px, "stop": s["stop"], "target": None},
                                60, "買い" if fresh else "保有中の目安", sent=True, conditions="流れ買い")
                positions.add(1, s["entry"], s["stop"], s["entry"] + 50.0, KEY, int(s["entry_time"].timestamp()))
            out.append(("ready", title, body))
        else:
            p = mine[-1]
            if s["stop"] > p["stop"] + 0.10:      # 10pips 以上上げた時だけ知らせる
                old = p["stop"]
                for q_ in ps:
                    if q_["id"] == p["id"]:
                        q_["stop"] = round(s["stop"], 3)
                if not dry:
                    positions.save(ps)
                if s["stop"] - old >= 0.20 or old < p["entry"] <= s["stop"]:
                    out.append(("info", f"🔼 流れ買い: 損切りの引き上げ USD/JPY {old:.3f} → {s['stop']:.3f}",
                                f"最も伸びた値 {s['best']:.3f} から {TRAIL_ATR:.0f}×日足ATR 後ろ。今 {px:.3f}。逆指値の注文を直してください"))
    else:
        if mine:      # 研究の関数では決済済み (損切り or 流れの反転)。5分監視の損切りで先に閉じていなければ知らせる
            p = mine[-1]
            last = s["last"]
            if not dry:
                positions._record(p, px, "流れ買いの決済", int(pd.Timestamp.now(tz="UTC").timestamp()))
                positions.save([q_ for q_ in ps if q_["id"] != p["id"]])
            out.append(("wait", f"🔔 流れ買い: 決済 USD/JPY 今 {px:.3f}",
                        f"流れが下向きに変わったか、損切りに届きました ({(last.exit_time + pd.Timedelta(hours=9)):%m/%d %H:%M})。"
                        f"入った値 {p['entry']:.3f} → 今 {px:.3f} ({(px - p['entry']) * 100:+.0f}pips)"))
    return out
