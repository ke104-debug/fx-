"""AI_FX の3つの強み (すべて手元の計算で判定。追加料金なし)

  急騰・急落検知  : 普段の値動き (直近20日の1時間ごとの変動) と比べて、明らかに異常な動きを検知する
  高確度局面検知  : 複数の条件 (時間足の方向・重要な価格帯・AIモデル・市場心理・相場状態・作戦シグナル) が一致した場面を探す
  ノートレード判定: 勝ちにくい場面 (重要指標の直前、薄商い、方向感のなさ、急変の最中など) では取引を推奨しない
"""
from datetime import timedelta

import numpy as np
import pandas as pd

from . import strategies as st

JP_HOLIDAYS_2026 = {"2026-11-23", "2026-12-31", "2027-01-01", "2027-01-11"}  # 東京が薄くなる日 (主なもの)
US_HOLIDAYS_2026 = {"2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18"}


def anomaly(h: pd.DataFrame) -> dict | None:
    """直近の値動きが普段の何倍か (σ = 1時間の値動きの標準偏差, 直近20日)"""
    r = h.Close.diff()
    sd = r.rolling(480).std().iat[-1]
    m1 = r.iat[-1]
    m3 = h.Close.iat[-1] - h.Close.iat[-4]
    m24 = h.Close.iat[-1] - h.Close.iat[-25]
    z1, z3, z24 = m1 / sd, m3 / (sd * np.sqrt(3)), m24 / (sd * np.sqrt(24))
    worst = max((abs(z1), "1時間", m1), (abs(z3), "3時間", m3), (abs(z24), "24時間", m24), key=lambda x: x[0])
    level = "急変" if worst[0] >= 4 or abs(m24) >= 2.0 else "注意" if worst[0] >= 3 else None
    if not level:
        return {"level": None, "z": round(worst[0], 1)}
    return {"level": level, "span": worst[1], "move": round(worst[2], 2), "z": round(worst[0], 1),
            "text": f"{worst[1]}で{worst[2]:+.2f}円 (普段の{worst[0]:.1f}倍の動き)"}


def no_trade(h: pd.DataFrame, now: pd.Timestamp, regime: str, tf_trends: dict, anom: dict) -> list[str]:
    """取引を見送るべき理由のリスト (空なら問題なし)"""
    why = []
    from . import calendar as cal
    live = now > pd.Timestamp.now(tz="UTC") - timedelta(days=6)   # カレンダーは今週分だけなので、過去の検証では使わない
    hi = cal.upcoming(now, 6, ("High",)) if live else []
    if hi:
        e = hi[0]
        why.append(f"重要指標 {e['emoji']} {e['title']} まで{(e['time'] - now).total_seconds() / 3600:.0f}時間 (日本時間 {e['time'].tz_convert('Asia/Tokyo'):%H:%M})")
    elif live and (md := cal.upcoming(now, 1, ("Medium",))):
        why.append(f"指標 {md[0]['emoji']} {md[0]['title']} まで1時間以内")
    else:
        ev = st.events(h.index[-50:].append(pd.DatetimeIndex([now + timedelta(days=10)])))
        nxt = ev[ev > now]
        if len(nxt) and (nxt[0] - now) <= timedelta(hours=6):
            why.append(f"重要指標 (FOMC/米雇用統計) まで{(nxt[0] - now).total_seconds() / 3600:.0f}時間")
    jst = now.tz_convert("Asia/Tokyo")
    if f"{jst:%Y-%m-%d}" in JP_HOLIDAYS_2026 and jst.hour < 16:
        why.append("日本の祝日で東京時間の取引が薄い")
    if f"{now:%Y-%m-%d}" in US_HOLIDAYS_2026:
        why.append("米国の祝日で取引が薄い")
    if now.weekday() == 4 and now.hour >= 19 or now.weekday() == 5:
        why.append("週末の手前で、週明けに窓を開けるリスク")
    ny_h = now.tz_convert("America/New_York").hour
    if 16 <= ny_h < 19:   # ニューヨーク市場の終わり (17時) の前後。夏時間で日本時間がずれるので NY 時間で判定
        why.append("ニューヨーク終盤〜早朝はスプレッドが広がりやすい時間帯")
    if anom.get("level") == "急変":
        why.append("急変の最中 (下げ止まり・上げ止まりの確認前)")
    dirs = {v for v in tf_trends.values() if v != "もみ合い"}
    if len(dirs) > 1:
        why.append("時間足どうしの方向が逆 (" + ", ".join(f"{k}:{v}" for k, v in tf_trends.items()) + ")")
    rng24 = h.High.iloc[-24:].max() - h.Low.iloc[-24:].min()
    if rng24 < 0.25:
        why.append(f"値幅が小さすぎる (24時間で{rng24:.2f}円)。コスト負けしやすい")
    return why


# 信頼度 (0〜100) の配点。2024-26年のリプレイ検証で、前半だけを使って決め、後半で効果を確認したもの:
#   重要な価格帯の近く → 前半・後半とも成績が良い (+30)
#   時間足の方向がそろっている → 前半・後半とも成績が悪い (動ききった後の飛び乗りになりやすい) (-15)
#   AIモデルの確率が同じ向き → 前半・後半とも成績が悪い (-10)
#   市場心理・相場状態・まとめ役の賛否 → 前半と後半で効果が逆。点数には入れず、通知の理由・注意として表示だけする
# 自己改善で journal/learning.json の "cond_weights" に上書きされることがある
COND_WEIGHTS = {"作戦シグナル": 45, "価格帯": 30, "時間足": -15, "AIモデル": -10, "市場心理": 0, "相場状態": 0}
BASE = 25
SEND_AT, HIGH_AT = 65, 80   # 毎時の分析: この点数以上で通知 / 高確度 (リプレイで55〜64%の帯は前半マイナスのため65に)
SEND_AT_QUICK = 55          # 5分監視 (下値・上値での反発確認): リプレイで55以上の通知が見送りより成績が良かった


def confidence(ok: list[str], coord_agree: int = 0, penalties: int = 0, weights: dict | None = None) -> int:
    """基本点 + そろった条件の配点 - 警戒事項 (1つ10点)。0〜100。
    まとめ役の賛否は検証で効果が一定しなかったため点数に入れない (引数は互換のため残す)"""
    w = weights or COND_WEIGHTS
    pts = BASE + sum(v for k, v in w.items() if any(x.startswith(k) for x in ok))
    return int(np.clip(pts - 10 * penalties, 0, 100))


def confluence(side: int, tf_trends: dict, near_level: str | None, p_up: float, vix_change: float | None,
               regime: str, has_signal: bool) -> tuple[int, list[str], list[str]]:
    """買い(side=1)/売り(-1) について、そろっている条件とそろっていない条件を数える"""
    want = "上昇" if side == 1 else "下降"
    ok, ng = [], []
    agree = [k for k, v in tf_trends.items() if v == want]
    against = [k for k, v in tf_trends.items() if v not in (want, "もみ合い")]
    desc = (f"{'・'.join(agree)}が{want}" if agree else f"{want}の時間足なし") + (f"、{'・'.join(against)}が逆" if against else "")
    (ok if agree and not against else ng).append(f"時間足 ({desc})")
    (ok if near_level else ng).append(f"価格帯 ({near_level or '重要な価格帯が近くにない'})")
    prob = p_up if side == 1 else 1 - p_up
    (ok if prob >= 0.58 else ng).append(f"AIモデル ({'上昇' if side == 1 else '下落'}確率 {prob:.0%}、ただし検証の正解率は約51〜54%)")
    if vix_change is not None:
        calm = vix_change < 0
        (ok if (calm and side == 1) or (not calm and side == -1) else ng).append(
            f"市場心理 (VIX {'低下=リスク回避が和らぐ' if calm else '上昇=リスク回避'})")
    (ok if regime in ("通常", "トレンド") else ng).append(f"相場状態 ({regime})")
    (ok if has_signal else ng).append("作戦シグナル (検証済みの作戦" + ("が発生)" if has_signal else "なし)"))
    return len(ok), ok, ng
