"""組織の競争リーグ (利用者の提案 2026-11-27: ほぼ同じ組織をいくつも作って成果を競わせる)。
今の本隊 (通知している『流れ×タイミング』) と、少しずつ違う挑戦者の組織が、同じ15分足で紙の上の取引を記録する (通知しない・AI は使わない)。
入れ替えの決まり (偶然の勝ちを選ばないため): 挑戦者が 100回以上の取引で本隊を上回り、その差が偶然では説明できない (t ≥ 2) こと、
23年の検証でも作る・確かめるの両方でプラスであること → ボスAIが提案し、利用者が承認して入れ替える。
記録: journal/league.csv (入った時刻・組織・向き・値段・ATR・結果 R)。結果は夜の検証部屋で Yahoo の15分足から決済をたどって埋める。"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
LOG = os.path.join(ROOT, "journal", "league.csv")
CHAMP = "本隊 (3日+半日・追いかけ)"
# 組織: 流れの長さ (日)・決済・金利の向きを見るか。23年の成績 (Dukascopy 15分足・0.2pips・次の足で入る、2026-11-27 確認) はどれも作る・確かめる両方でプラス
ORGS = {
    CHAMP: dict(big=3.0, mid=0.5, exit="追いかけ", fund=False, r23="+0.113R (作る+0.08/確かめ+0.15)"),
    "長めの流れ (5日+1日)": dict(big=5.0, mid=1.0, exit="追いかけ", fund=False, r23="+0.111R (作る+0.08/確かめ+0.15)"),
    "短めの流れ (2日+半日)": dict(big=2.0, mid=0.5, exit="追いかけ", fund=False, r23="+0.108R (作る+0.08/確かめ+0.14)"),
    "金利の向きも見る (3日+半日)": dict(big=3.0, mid=0.5, exit="追いかけ", fund=True, r23="+0.111R (作る+0.11/確かめ+0.11・回数は半分)"),
    "利確を固定 (3日+半日・損1:利2)": dict(big=3.0, mid=0.5, exit="固定", fund=False, r23="+0.052R (0.8pipsだとマイナス)"),
    "日足の流れ (20日+1日)": dict(big=20.0, mid=1.0, exit="追いかけ", fund=False, r23="+0.076R (0.8pipsだとマイナス)"),
}
EXITS = {"追いかけ": dict(sl=1.0, tp=0, trail=2.0, be=1.0, hold=96), "固定": dict(sl=1.0, tp=2.0, trail=0, be=0, hold=48)}


def _us2_dir() -> int:
    try:
        from .org.data import _fred
        s = _fred("DGS2")
        return 1 if s.iat[-2] - s.iat[-22] > 0 else -1
    except Exception:
        return 0


def record(d: pd.DataFrame, now: pd.Timestamp):
    """live_tl から15分足の確定ごとに呼ぶ。各組織の最後の足の合図を記録 (通知しない)"""
    from . import flowtiming as ft
    jh = (now + pd.Timedelta(hours=9)).hour
    if not (jh >= 10 or jh < 1):
        return []
    t = d.index[-1]
    rows, us2 = [], None
    for name, o in ORGS.items():
        s = int(ft.trendline_signal(d.iloc[-1500:], big=o["big"], mid=o["mid"])[-1])
        if not s:
            continue
        if o["fund"]:
            us2 = _us2_dir() if us2 is None else us2
            if us2 and us2 != s:
                continue
        rows.append({"time_utc": f"{t:%Y-%m-%d %H:%M}", "org": name, "side": s, "close": round(float(d.Close.iat[-1]), 3),
                     "atr": round(float(d.atr.iat[-1]), 4), "R": ""})
    if rows:
        old = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame()
        if len(old):
            have = set(zip(old.time_utc, old.org))
            rows = [r for r in rows if (r["time_utc"], r["org"]) not in have]
        if rows:
            os.makedirs(os.path.dirname(LOG), exist_ok=True)
            pd.concat([old, pd.DataFrame(rows)]).to_csv(LOG, index=False)
    return rows


def settle(bars: pd.DataFrame | None = None) -> int:
    """結果の出ていない紙の取引を、15分足で決済までたどる (研究と同じ決済: explore.run)"""
    if not os.path.exists(LOG):
        return 0
    from . import explore as ex, live_tl
    d = pd.read_csv(LOG)
    todo = d[d.R.isna() | (d.R.astype(str) == "")]
    if todo.empty:
        return 0
    raw = bars if bars is not None else live_tl._bars15()
    if raw is None or raw.empty:
        return 0
    b = ex.bars(tf=None, src=raw.copy())
    b = b[b.atr > 0]
    O, H, L, C, A = (b[k].values.astype(np.float64) for k in ("Open", "High", "Low", "Close", "atr"))
    n = 0
    for i, r in todo.iterrows():
        t = pd.Timestamp(r.time_utc, tz="UTC")
        if t not in b.index:
            continue
        k = b.index.get_loc(t)
        e = EXITS[ORGS.get(r.org, ORGS[CHAMP])["exit"]]
        if k + e["hold"] + 1 >= len(b):
            continue          # まだ決済まで時間がたっていない
        sig = np.zeros(len(b), np.int64); sig[k] = int(r.side)
        _, _, _, R = ex.run(O, H, L, C, A, sig, np.ones(len(b), bool), e["sl"], e["tp"], e["trail"], e["be"], e["hold"], 0.002, False)
        if len(R):
            d.loc[i, "R"] = round(float(R[0]), 3); n += 1
    d.to_csv(LOG, index=False)
    return n


def table() -> list[str]:
    if not os.path.exists(LOG):
        return ["組織の競争リーグ: まだ記録なし"]
    d = pd.read_csv(LOG)
    d["R"] = pd.to_numeric(d.R, errors="coerce")
    done = d.dropna(subset=["R"])
    out = []
    champ = done[done.org == CHAMP]
    for name, o in ORGS.items():
        g = done[done.org == name]
        line = f"{'👑' if name == CHAMP else '・'} {name}: {len(g)}回 平均 {g.R.mean() if len(g) else 0:+.2f}R 合計 {g.R.sum():+.1f}R (23年 {o['r23']})"
        if name != CHAMP and len(g) >= 100 and len(champ) >= 100:
            diff = g.R.mean() - champ.R.mean()
            se = np.sqrt(g.R.var() / len(g) + champ.R.var() / len(champ))
            t = diff / se if se else 0
            line += f" / 本隊との差 {diff:+.3f}R (t={t:.1f})" + (" → ★入れ替えの候補 (ボスAIが提案)" if t >= 2 else "")
        out.append(line)
    out.append(f"(記録 {len(d)}件・結果あり {len(done)}件。入れ替えは100回以上・本隊との差が t≥2・23年でも両期間プラスの時だけ)")
    return out
