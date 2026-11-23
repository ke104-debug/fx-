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

from . import edge, notify
from . import portfolio as pf
from . import regime as rg

CAPITAL = float(os.getenv("FX_CAPITAL", "1000000"))
STATE = os.getenv("FX_SCAN_STATE", "state/scan.json")
JST = "Asia/Tokyo"

# (作戦, 相場状態) -> 1回のリスク (資金に対する損失上限)。最大下落20%以内を目安に抑えた設定
RULES = {("S", "荒れ"): 0.04, ("H4", "通常"): 0.015, ("H5", "通常"): 0.01, ("H3", "通常"): 0.01, ("H3", "トレンド"): 0.015}
LABEL = {"S": "急変の反動 (流れの方向)", "H3": "東京レンジのロンドン抜け", "H4": "レンジ逆張り (流れの方向)", "H5": "指標発表直後の順張り"}
EXIT_TEXT = {"eod": "届かなければ当日 日本時間朝5時に手仕舞い", "24h": "届かなければ24時間後に手仕舞い", None: "損切りか利確まで保有 (最長48時間)"}
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


def run(dry=False):
    h = rg.prepare(load_recent())
    n = len(h) - 1
    t = h.index[n]
    state_now = rg.NAMES[h.regime.iat[n]]
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    msgs = []
    # 相場状態の変化
    if st.get("regime") != state_now and (state_now == "荒れ" or awake(pd.Timestamp.now(tz="UTC"))):
        msgs.append(("info", f"相場の状態: {st.get('regime', '-')} → {state_now}",
                     f"USD/JPY {h.Close.iat[n]:.2f}。使う作戦: " + ", ".join(LABEL[s] for (s, r) in RULES if r == state_now)))
        st["regime"] = state_now
    # 急変の予告
    anom = edge.anomaly(h)
    if anom.get("level") and st.get("alert_key") != f"{anom['level']}-{t:%Y-%m-%d}" and (anom["level"] == "急変" or awake(pd.Timestamp.now(tz="UTC"))):
        msgs.append(("wait", f"{'急騰' if anom['move'] > 0 else '急落'}を検知 ({anom['level']})",
                     f"USD/JPY {h.Close.iat[n]:.2f}円。{anom['text']}。\n"
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
        conf = "高" if n_ok >= 5 else "中" if n_ok == 4 else "低"
        verdict = "ノートレード推奨" if serious else f"{'買い' if s['side'] == 1 else '売り'}"
        tgt = f"{s['target']:.2f}" if s.get("target") else "なし (追いかけ決済)"
        price = h.Close.iat[n]
        body = (f"【現在の相場】{price:.2f}円 / 24時間 {h.Low.iloc[-24:].min():.2f}〜{h.High.iloc[-24:].max():.2f} / 状態: {state_now}\n\n"
                f"【判定】{verdict}" + (f" ({'・'.join(serious)})" if serious else "") + "\n"
                f"【信頼度】{conf} (条件 {n_ok}/{n_ok + len(ng)} 一致)\n\n"
                f"【判断の理由】\n" + "\n".join("・" + x for x in ok) + "\n"
                f"【そろっていない条件】\n" + ("\n".join("・" + x for x in ng) or "・なし") + "\n\n"
                f"【売買戦略】{LABEL[key]}\n"
                f"{'買い' if s['side'] == 1 else '売り'} 目安 {s['entry']:.2f} / 損切り {s['stop']:.2f} / 利確 {tgt}\n"
                f"{s['note']}\n\n"
                f"【警戒事項】\n" + ("\n".join("・" + x for x in stop_why) or "・特になし") + "\n\n"
                f"足 {t.tz_convert(JST):%m/%d %H:%M} 確定時点。※過去検証の作戦であり利益を保証しません")
        title = ("ノートレード推奨: " if serious else "高確度シグナル: " if conf == "高" else "売買シグナル: ") + LABEL[key]
        msgs.append(("wait" if serious else "ready", title, body))
        st.setdefault("sent", []).append(sig_id)
        st["sent"] = st["sent"][-200:]
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
