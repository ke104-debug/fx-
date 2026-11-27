"""合図の点検 (2026-11-27 の不具合の再発防止)。
11/27 17:22、5分監視の道だけ流れの確認が抜けていて、上昇の流れの中で「レンジ逆張り (流れの方向)」の売りを出した。利用者は入って損切り。
1. 送る前の点検 `rule_ok`: 合図が、その作戦の検証済みの条件 (名前どおりの条件) を満たしているか。満たさなければ送らない
2. 送った後の点検 `review_open`: 見張り中の合図を毎回点検し、入った時点で条件を満たしていなかったと分かったら
   「⚠️ 合図の取り消し」を1回だけ送り、見張りを閉じる (記録には『取り消し』として残す)
ルールそのものは変えない。名前と中身の食い違いを止めるためのもの。"""
import pandas as pd

# 作戦ごとの「必ず満たすべき条件」(検証済みの形)。流れの方向 = 大きな流れ (20日平均の向き、trend_up) と同じ向き
WITH_TREND = {"H4": "レンジ逆張り (流れの方向)", "S": "急変の反動 (流れの方向)", "SF": "急変に付いていく (流れの方向)"}


def rule_ok(key: str, side: int, h: pd.DataFrame, i: int | None = None) -> tuple[bool, str]:
    i = len(h) - 1 if i is None else i
    if key in WITH_TREND and "trend_up" in h:
        up = bool(h.trend_up.iat[i])
        if (side == 1) != up:
            return False, (f"{WITH_TREND[key]}の合図なのに、大きな流れ (20日平均) が{'上向き' if up else '下向き'}で"
                           f"{'売り' if side == -1 else '買い'}になっている")
    return True, ""


def review_open(h: pd.DataFrame, dry=False) -> list[tuple[str, str, str]]:
    """見張り中の合図 (利用者が自分で登録したものは除く) を点検。食い違いがあれば取り消しの通知を作り、見張りを閉じる"""
    from . import positions
    ps = positions.load()
    out, keep = [], []
    now_px = float(h.Close.iat[-1])
    for p in ps:
        if p.get("key") == "U":
            keep.append(p); continue
        t = pd.Timestamp(p["opened"], unit="s", tz="UTC") - pd.Timedelta(hours=1)
        j = h.index.searchsorted(t, side="right") - 1
        if j < 0:
            keep.append(p); continue
        ok, why = rule_ok(p["key"], p["side"], h, j)
        if ok:
            keep.append(p); continue
        pips = p["side"] * (now_px - p["entry"]) * 100
        word = "売り" if p["side"] == -1 else "買い"
        body = (f"**{word} {p['entry']:.3f} の合図を取り消します (システムの誤り)**\n"
                f"理由: {why}。検証済みの作戦の条件を満たしていませんでした\n"
                f"今 {now_px:.3f} ({pips:+.0f}pips)\n━━━━━━━━━━\n"
                "・この合図で入っている場合は、決済を検討してください (この合図は作戦の根拠がありません)\n"
                "・損切りの注文 ({:.3f}) が残っていれば、決済するまでは消さないでください\n"
                "・見張りはここで終えます".format(p["stop"]))
        out.append(("wait", f"⚠️ 合図の取り消し USD/JPY {word} {p['entry']:.3f}", body))
        if not dry:
            positions._record(p, now_px, "取り消し (作戦の条件の食い違い)", int(pd.Timestamp.now(tz="UTC").timestamp()))
    if out and not dry:
        positions.save(keep)
    return out


# ---------------------------------------------------------------- 監査AI・点検部隊の指摘を判断に使う (2026-11-27 利用者の指摘: 意見を言うだけで参考にされていない)
CRITICAL = ("合図と作戦の条件", "損切り・利確の向き", "見張り中の取引", "通知している作戦の根拠")   # 売買の判断に関わるものだけ (記録の遅れでは止めない)


def inspector_block(hours=2) -> list[str]:
    """点検部隊の直近の結果で、売買の判断に関わる問題 (CRITICAL) が NG のままなら、その内容を返す"""
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "journal", "inspector.csv")
    try:
        d = pd.read_csv(path)
    except Exception:
        return []
    d = d[pd.to_datetime(d.time_utc, utc=True) >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)]
    if d.empty:
        return []
    last = d.groupby("check").tail(1)
    bad = last[(last.check.isin(CRITICAL)) & (last.ok.astype(str) != "True")]
    return [f"{r.check}: {str(r.detail)[:80]}" for r in bad.itertuples()]


def audit_notes(hours=1.5) -> list[str]:
    """監査AIの直近の指摘 (毎時の会議の記録の『監査』欄)"""
    import os
    try:
        o = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "journal", "org.csv"), usecols=["time_utc", "監査"])
    except Exception:
        return []
    o = o[pd.to_datetime(o.time_utc, utc=True) >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)].dropna()
    if o.empty:
        return []
    return [x.strip() for x in str(o["監査"].iat[-1]).split(";") if x.strip()]


def gate(side: int) -> tuple[bool, list[str]]:
    """合図を送る前の最終関門。(止める?, 通知に書く注意)。
    - 点検部隊が売買の判断に関わる問題を見つけたまま → 止める
    - 監査AIの指摘 → 通知の注意に必ず書く。『1人の意見でほぼ決まっている』会議の向きと逆の合図は注意を強める"""
    blk = inspector_block()
    if blk:
        return True, ["点検部隊が問題を検出中のため送らない: " + " / ".join(blk)]
    notes = [f"監査AI: {x}" for x in audit_notes()]
    return False, notes
