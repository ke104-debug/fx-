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

from . import notify
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


def load_recent() -> pd.DataFrame:
    d = yf.download("USDJPY=X", period="120d", interval="1h", progress=False, auto_adjust=False)
    d.columns = d.columns.get_level_values(0)
    d = d[["Open", "High", "Low", "Close"]].dropna()
    d.index = pd.to_datetime(d.index, utc=True)
    now = pd.Timestamp.now(tz="UTC")
    return d[d.index + pd.Timedelta(hours=1) <= now]  # 確定した足だけ


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
    if st.get("regime") != state_now:
        msgs.append(("info", f"相場の状態: {st.get('regime', '-')} → {state_now}",
                     f"USD/JPY {h.Close.iat[n]:.2f}。使う作戦: " + ", ".join(LABEL[s] for (s, r) in RULES if r == state_now)))
        st["regime"] = state_now
    # 急変の予告
    al = shock_alert(h)
    if al and st.get("alert_day") != f"{t:%Y-%m-%d}":
        msgs.append(("wait", "急変を検知 (まだ入らない)", al + "。下げ止まり (上げ止まり) を確認したら売買シグナルを送ります。原因の確認をおすすめします。"))
        st["alert_day"] = f"{t:%Y-%m-%d}"
    # 売買シグナル
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
    for key, s in sigs:
        sig_id = f"{key}-{t:%Y%m%d%H}"
        if sig_id in st.get("sent", []):
            continue
        risk = RULES[(key, state_now)]
        units, loss = size(s["entry"], s["stop"], risk)
        tgt = f"{s['target']:.2f}" if s.get("target") else "なし (追いかけ決済)"
        body = (f"**{'買い' if s['side'] == 1 else '売り'}** USD/JPY 目安 {s['entry']:.2f}\n"
                f"損切り {s['stop']:.2f} / 利確 {tgt}\n"
                f"数量の目安: {units:,.0f}ドル (資金{CAPITAL / 1e4:.0f}万円・損切り時の損失 約{loss / 1e4:.1f}万円 = {risk:.1%})\n"
                f"相場状態: {state_now} / {s['note']}\n"
                f"足 {t.tz_convert(JST):%m/%d %H:%M} 確定時点。※過去検証の作戦であり利益を保証しません")
        msgs.append(("ready", f"売買シグナル: {LABEL[key]}", body))
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
