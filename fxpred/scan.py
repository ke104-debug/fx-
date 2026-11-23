"""毎時の監視: 相場の状態を判定し、その状態で使う作戦のシグナルが出た時だけ Discord に通知する。
python -m fxpred scan  (GitHub Actions から毎時実行。状態は state/scan.json に保存して重複通知を防ぐ)

運用ルール (RULES) は 2024-2026 年の検証で決めたもの。20年データでの確認後に見直す。
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from . import edge, fmt, notify, specialists as sp
from . import portfolio as pf
from . import regime as rg

CAPITAL = float(os.getenv("FX_CAPITAL", "1000000"))
STATE = os.getenv("FX_SCAN_STATE", "state/scan.json")
JST = "Asia/Tokyo"

# (作戦, 相場状態) -> 1回のリスク (資金に対する損失上限)。最大下落20%以内を目安に抑えた設定
RULES = {("S", "荒れ"): 0.04, ("H4", "通常"): 0.015, ("H5", "通常"): 0.01, ("H3", "通常"): 0.01, ("H3", "トレンド"): 0.015}
LABEL = {"S": "急変の反動 (流れの方向)", "H3": "東京レンジのロンドン抜け", "H4": "レンジ逆張り (流れの方向)", "H5": "指標発表直後の順張り"}
EXIT_TEXT = {"eod": "届かなければ当日 日本時間朝5時に手仕舞い", "24h": "届かなければ24時間後に手仕舞い", None: "損切りか利確まで保有 (最長48時間)"}
EXIT_VALID = {"H3": "当日 朝5時まで", "H4": "最長48時間", "H5": "24時間", "S": "最長72時間"}
RULE_FN = {"H3": pf.h_tokyo_break, "H4": pf.h_range_trend, "H5": pf.h_event_momo}


CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "notify.json")
URGENT = {"S"}   # 時間外でも送る作戦 (急変の反動)


def awake(now_utc: pd.Timestamp, events=None) -> bool:
    """利用者が起きている時間か (config/notify.json)。指標発表の前後と、日付ごとの追加時間も含める"""
    cfg = json.load(open(CONFIG)) if os.path.exists(CONFIG) else {"start": "10:00", "end": "25:00"}
    jst = now_utc.tz_convert(JST)
    def minutes(hhmm): h, m = map(int, hhmm.split(":")); return h * 60 + m
    # 0〜1時台は「前日の24〜25時」として扱う
    day = (jst - pd.Timedelta(hours=1)).normalize() if jst.hour < 1 else jst.normalize()
    cur = (jst - day).total_seconds() / 60
    start = minutes(cfg.get("start", "10:00"))
    for ex in cfg.get("extra", []):
        if ex.get("date") == f"{day:%Y-%m-%d}":
            start = minutes(ex.get("start", cfg["start"]))
    if start <= cur < minutes(cfg.get("end", "25:00")):
        return True
    if events is not None and len(events):
        lo, hi = cfg.get("event_window_hours", [-1, 3])
        dt = (now_utc - events).total_seconds() / 3600
        if ((dt >= lo) & (dt <= hi)).any():
            return True
    return False


def load_recent() -> pd.DataFrame:
    d = yf.download("USDJPY=X", period="120d", interval="1h", progress=False, auto_adjust=False)
    d.columns = d.columns.get_level_values(0)
    d = d[["Open", "High", "Low", "Close"]].dropna()
    d.index = pd.to_datetime(d.index, utc=True)
    now = pd.Timestamp.now(tz="UTC")
    return d[d.index + pd.Timedelta(hours=1) <= now]  # 確定した足だけ


def context(h: pd.DataFrame) -> dict:
    """通知に使う判断材料: 時間足の方向、AIモデルの上昇確率 (4時間後)、VIX の4時間変化"""
    from . import diagnose, model, features as ft
    raw = h[["Open", "High", "Low", "Close"]]
    tech = diagnose.technical(raw)
    tf = {k: tech[k]["trend"] for k in ("15分足", "1時間足", "4時間足")}
    try:
        f = ft.build(raw, "USDJPY=X", np.log(raw.Close).diff(), horizon=4)
        m = model.train(f)
        p_up = float(model.predict_proba(m, f.tail(1)).iloc[0])
    except Exception:
        p_up = 0.5
    try:
        v = yf.download("^VIX", period="5d", interval="1h", progress=False)["Close"].squeeze().dropna()
        vix_chg = float(v.iat[-1] - v.iat[-5])
    except Exception:
        vix_chg = None
    return {"tf": tf, "p_up": p_up, "vix_chg": vix_chg, "tech": tech}


def shock_signal(h: pd.DataFrame, drop=2.0, lookback=24, wait=12, retrace=0.4, buffer=0.2):
    """最新の確定足が「急変後の下げ止まり (上げ止まり)」の初回なら、次の足で入るシグナルを返す"""
    n = len(h) - 1
    up = bool(h.trend_up.iat[n])
    side = 1 if up else -1   # 流れの方向に戻る反動だけ
    H, L, C, O = h.High.values, h.Low.values, h.Close.values, h.Open.values
    for i in range(n - wait, n):          # 直近 wait 時間以内に急変を検知したか
        top = H[i - lookback:i + 1].max() if side == 1 else L[i - lookback:i + 1].min()
        mv = (top - C[i]) if side == 1 else (C[i] - top)
        if mv < drop:
            continue
        ext = L[i] if side == 1 else H[i]
        for k in range(i + 1, n + 1):
            ext = min(ext, L[k]) if side == 1 else max(ext, H[k])
            turned = (C[k] > O[k] and L[k] > L[k - 1]) if side == 1 else (C[k] < O[k] and H[k] < H[k - 1])
            if turned:
                if k != n:
                    return None   # もっと前に下げ止まり済み (すでに通知対象)
                size = abs(top - ext)
                return dict(side=side, entry=C[n], stop=ext - side * buffer, target=ext + side * retrace * size,
                            note=f"24時間で{size:.2f}円の{'急落' if side == 1 else '急騰'}後、{'下げ' if side == 1 else '上げ'}止まりを確認")
        return None
    return None


def shock_alert(h: pd.DataFrame, drop=2.0):
    """下げ止まり前の予告: 急変を検知した時点で知らせる"""
    n = len(h) - 1
    hi, lo = h.High.iloc[n - 24:n + 1].max(), h.Low.iloc[n - 24:n + 1].min()
    c = h.Close.iat[n]
    if hi - c >= drop: return f"24時間で{hi - c:.2f}円の急落 (高値{hi:.2f}→{c:.2f})"
    if c - lo >= drop: return f"24時間で{c - lo:.2f}円の急騰 (安値{lo:.2f}→{c:.2f})"
    return None


def log_signal(t, key, s, conf, verdict, sent=True, conditions=""):
    """シグナルを journal/signals.csv に記録 (送らなかったものも。週次・月次の見直しで「見送りが正しかったか」も確かめる)"""
    path = os.path.join(os.path.dirname(__file__), "..", "journal", "signals.csv")
    row = {"time_utc": f"{t:%Y-%m-%d %H:%M}", "strategy": LABEL[key], "side": s["side"], "entry": round(s["entry"], 3),
           "stop": round(s["stop"], 3), "target": round(s["target"], 3) if s.get("target") else "", "confidence": conf, "verdict": verdict,
           "key": key, "sent": sent, "conditions": conditions}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def near_level(h: pd.DataFrame, side: int) -> str | None:
    """買いなら直近の安値 (下値)、売りなら直近の高値 (上値) の近く (0.25円以内) にいるか"""
    c = h.Close.iat[-1]
    for hours, name in ((24, "24時間"), (120, "5日")):
        lv = h.Low.iloc[-hours:].min() if side == 1 else h.High.iloc[-hours:].max()
        if abs(c - lv) <= 0.25:
            return f"{name}の{'安値' if side == 1 else '高値'} {lv:.2f} の近く"
    return None


def size(entry, stop, risk):
    loss_yen = CAPITAL * risk
    units = min(loss_yen / abs(entry - stop), pf.MAX_LEV * CAPITAL / entry)
    return units, loss_yen


def quick_chance(trigger, price):
    """簡易監視がチャンス候補を見つけた時の詳しい分析。信頼度が基準以上なら (タイトル, 本文, 色) を返す"""
    from .dailyplan import load_learning
    kind, level = trigger
    h = rg.prepare(load_recent())
    n = len(h) - 1
    now = pd.Timestamp.now(tz="UTC")
    state_now = rg.NAMES[h.regime.iat[n]]
    rv = sp.range_view(h, h.rsi.iat[n])
    if kind == "near_support":
        side, stop, tgt, why = 1, rv["support"] - 0.20, rv["resistance"] - 0.10, f"下値 {rv['support']:.2f} に接近"
    elif kind == "near_resistance":
        side, stop, tgt, why = -1, rv["resistance"] + 0.20, rv["support"] + 0.10, f"上値 {rv['resistance']:.2f} に接近"
    else:
        up = bool(h.trend_up.iat[n])
        side = 1 if price > level else -1
        if (side == 1) != up:
            return None   # 大きな流れと逆方向の反動は取らない (検証で成績が悪い)
        size = abs(h.Close.iloc[-24:].max() - h.Close.iloc[-24:].min())
        stop, tgt, why = level - side * 0.20, level + side * 0.4 * size, "急変後の下げ止まり (上げ止まり) を確認"
    if (tgt - price) * side <= 0.05 or (price - stop) * side <= 0.05:
        return None
    if kind != "rebound" and not awake(now, rg.st.events(h.index[-200:])):
        return None
    ctx = context(h)
    anom = edge.anomaly(h)
    n_ok, ok, ng = edge.confluence(side, ctx["tf"], f"{why}", ctx["p_up"], ctx["vix_chg"], state_now, False)
    learn = load_learning()
    views = {"急変担当": sp.shock_view(h, anom, sp.headlines(hours=6)), "レンジ担当": rv, "トレンド担当": sp.trend_view(h)}
    coord = sp.coordinate(state_now, views, learn)
    agree = 1 if coord["verdict"] == ("買い寄り" if side == 1 else "売り寄り") else \
        -1 if coord["verdict"] == ("売り寄り" if side == 1 else "買い寄り") else 0
    stop_why = edge.no_trade(h, now, state_now, ctx["tf"], anom) if kind != "rebound" else []
    serious = [w for w in stop_why if w.startswith(("重要指標", "急変の最中", "週末"))]
    score = edge.confidence(ok, agree, len(stop_why), learn.get("cond_weights"))
    send_at = learn.get("send_at", edge.SEND_AT)
    key = "S" if kind == "rebound" else "H4"
    log_signal(now.floor("min"), key, {"side": side, "entry": price, "stop": stop, "target": tgt}, score,
               "見送り" if serious else "買い" if side == 1 else "売り", sent=score >= send_at and agree >= 0,
               conditions="|".join(x.split(" ")[0] for x in ok))
    if score < send_at or agree < 0:
        print(f"(簡易監視の候補 {why}: 信頼度{score}で見送り)")
        return None
    reasons = [why] + [x for x in ok if not x.startswith(("相場状態", "価格帯"))]
    title, body = fmt.signal(side, price, stop, tgt, score, reasons, [x for x in ng] + stop_why,
                             f"{now.tz_convert(JST):%m/%d %H:%M} から最長48時間", serious)
    return title, body, (16776960 if serious else 5763719)


def run(dry=False):
    h = rg.prepare(load_recent())
    n = len(h) - 1
    t = h.index[n]
    state_now = rg.NAMES[h.regime.iat[n]]
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    msgs = []
    # 相場状態の変化
    if st.get("regime") != state_now and (state_now == "荒れ" or awake(pd.Timestamp.now(tz="UTC"))):
        MEAN = {"通常": "レンジになりやすい。下値・上値の近くを待つ", "トレンド": "一方向に動きやすい。押し目・戻りを待つ",
                "荒れ": "急な動き。下げ止まり・上げ止まりを確認してから"}
        msgs.append(("info", f"相場の状態: {st.get('regime', '-')} → {state_now}",
                     f"USD/JPY {h.Close.iat[n]:.2f}　{MEAN[state_now]}"))
        st["regime"] = state_now
    # 急変の予告
    anom = edge.anomaly(h)
    if anom.get("level") and st.get("alert_key") != f"{anom['level']}-{t:%Y-%m-%d}" and (anom["level"] == "急変" or awake(pd.Timestamp.now(tz="UTC"))):
        msgs.append(("wait", f"⚠️ {'急騰' if anom['move'] > 0 else '急落'} USD/JPY {h.Close.iat[n]:.2f} ({anom['level']})",
                     f"USD/JPY {h.Close.iat[n]:.2f}円。{anom['text']}。\n\n【急変担当の見立て】\n" +
                     "\n".join("・" + x for x in sp.shock_view(h, anom, sp.headlines(hours=6))["lines"]) + "\n\n"
                     "まだ入らないでください。原因 (介入・指標・要人発言・リスク回避) の確認をおすすめします。\n"
                     "流れの方向へ戻る下げ止まり (上げ止まり) を確認したら、反動狙いのシグナルを送ります。"))
        st["alert_key"] = f"{anom['level']}-{t:%Y-%m-%d}"
    # 売買シグナル
    is_awake = awake(pd.Timestamp.now(tz="UTC"), rg.st.events(h.index[-200:]))
    sigs = []
    if ("S", state_now) in RULES:
        s = shock_signal(h)
        if s: sigs.append(("S", s))
    for key, fn in RULE_FN.items():
        if (key, state_now) not in RULES:
            continue
        r = fn(h, n)
        if r:
            side, stop, tgt, trail = r
            sigs.append((key, dict(side=side, entry=h.Close.iat[n], stop=stop, target=tgt, note=EXIT_TEXT.get(trail, "損切りか利確まで保有"))))
    is_awake = awake(pd.Timestamp.now(tz="UTC"), rg.st.events(h.index[-200:]))
    ctx = context(h) if (sigs or anom.get("level")) else None
    now = pd.Timestamp.now(tz="UTC")
    for key, s in sigs:
        sig_id = f"{key}-{t:%Y%m%d%H}"
        if sig_id in st.get("sent", []):
            continue
        if not is_awake and key not in URGENT:
            print(f"(時間外のため送らない) {LABEL[key]}")
            st.setdefault("sent", []).append(sig_id)
            continue
        level = near_level(h, s["side"])
        n_ok, ok, ng = edge.confluence(s["side"], ctx["tf"], level, ctx["p_up"], ctx["vix_chg"], state_now, True)
        stop_why = edge.no_trade(h, now, state_now, ctx["tf"], anom) if key not in URGENT else []
        serious = [w for w in stop_why if w.startswith(("重要指標", "急変の最中", "週末"))]
        views = {"急変担当": sp.shock_view(h, anom, sp.headlines(hours=12)), "レンジ担当": sp.range_view(h, h.rsi.iat[n]),
                 "トレンド担当": sp.trend_view(h)}
        from .dailyplan import load_learning
        learn = load_learning()
        coord = sp.coordinate(state_now, views, learn)
        lead = coord["lead"]
        agree = 1 if coord["verdict"] == ("買い寄り" if s["side"] == 1 else "売り寄り") else \
            -1 if coord["verdict"] == ("売り寄り" if s["side"] == 1 else "買い寄り") else 0
        score = edge.confidence(ok, agree, len(stop_why), learn.get("cond_weights"))
        conf = f"{score}" 
        verdict = "ノートレード推奨" if serious else f"{'買い' if s['side'] == 1 else '売り'}"
        # 理由: そろった条件を、利用者に意味のある言葉に (作戦名は出さない)
        reasons = [x for x in ok if not x.startswith(("作戦シグナル", "相場状態"))]
        reasons = [x.replace("AIモデル (", "AIモデル (").split("、ただし")[0].rstrip(")") + (")" if "(" in x else "") for x in reasons]
        if any(x.startswith("作戦シグナル") for x in ok):
            reasons.append("過去の検証で成績の良かった値動きのパターンが発生")
        lv = views[lead]["lines"][0] if lead == "レンジ担当" else None
        if lv: reasons.append(lv.split(" / レンジ内")[0])
        cautions = [x for x in ng if not x.startswith("作戦シグナル")] + stop_why
        exit_h = {"eod": "本日 朝5時まで", "24h": "24時間", None: "48時間"}
        valid = EXIT_VALID.get(key, "48時間")
        title, body = fmt.signal(s["side"], s["entry"], s["stop"], s.get("target"), score, reasons, cautions,
                                 f"{t.tz_convert(JST):%m/%d %H:%M} から{valid}", serious)
        send_at = learn.get("send_at", edge.SEND_AT)
        skip = (score < send_at or agree < 0) and key not in URGENT
        log_signal(t, key, s, score, verdict, sent=not skip, conditions="|".join(x.split(" ")[0] for x in ok))
        if skip:
            print(f"(チャンスと判断しないため送らない: 信頼度{score}{'、担当AIと逆向き' if agree < 0 else ''}) {LABEL[key]}")
            st.setdefault("sent", []).append(sig_id)
            continue
        msgs.append(("wait" if serious else "ready", title, body))
        st.setdefault("sent", []).append(sig_id)
        st["sent"] = st["sent"][-200:]
    try:
        rv = sp.range_view(h, h.rsi.iat[n])
        lv_path = os.path.join(os.path.dirname(STATE) or ".", "levels.json")
        json.dump({"support": rv["support"], "resistance": rv["resistance"], "regime": state_now,
                   "trend_up": bool(h.trend_up.iat[n]), "time": f"{t:%Y-%m-%d %H:%M}"}, open(lv_path, "w"), ensure_ascii=False)
    except Exception:
        pass
    st["last_run"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    st["last_bar"] = f"{t:%Y-%m-%d %H:%M}"
    st["price"] = round(float(h.Close.iat[n]), 3)
    for kind, title, body in msgs:
        print(f"[{kind}] {title}\n{body}\n")
        if not dry:
            color = {"ready": 5763719, "wait": 16776960, "info": 9807270}[kind]
            notify.send(title, body, color=color)
    if not msgs:
        print(f"{t:%Y-%m-%d %H:%M} UTC 状態={state_now} 価格={h.Close.iat[n]:.2f} シグナルなし")
    os.makedirs(os.path.dirname(STATE) or ".", exist_ok=True)
    json.dump(st, open(STATE, "w"), ensure_ascii=False, indent=1)
    return st
