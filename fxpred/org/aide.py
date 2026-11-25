"""統括補佐AI (Gemini 無料枠): 売買の合図を送る直前に、ルールでは拾えないことを最終確認する。
見るもの: 合図の中身・全統括AIの結論とリスク・直近のニュースの見出し・24時間以内の指標・保有中。
答え: 入ってよい / 慎重に / 見送るべき と理由 (40字)。

まだ「検証中」: 合図は止めない。通知に1行添えるだけで、答えを journal/aide.csv に残し、
「補佐AIが見送るべきと言った合図」と「入ってよいと言った合図」のその後の成績を週次で比べる。
見送るべきの方がはっきり成績が悪いと確かめられたら (30件以上)、提案として検証部隊に出し、合図を止める判断に使う。"""
import os

import pandas as pd

PATH = os.path.join(os.path.dirname(__file__), "..", "..", "journal", "aide.csv")


def check(side: int, entry: float, stop: float, target: float, strategy: str, reasons: list[str], cautions: list[str], h=None) -> str:
    """通知に添える1行を返す (呼べなければ空)"""
    try:
        from . import gem, meeting
        from .. import specialists as sp, calendar as cal
        res = meeting.final(h) if h is not None else {}
        heads = "\n".join(f"- {(x['time'] + pd.Timedelta(hours=9)):%H:%M} {x['title']}" for x in sp.headlines(hours=6)[:20])
        now = pd.Timestamp.now(tz="UTC")
        ev = " / ".join(cal.fmt(e, False) for e in cal.upcoming(now, 24, impacts=("High",))[:3]) or "なし"
        r = gem.call("統括補佐", "あなたは FX (ドル円) の売買判断チームの統括補佐。チームのルールが次の合図を出した。"
                     "ルールでは拾えない材料 (直近のニュース・指標の直前・介入の気配・急な材料) があるかを確かめ、"
                     "入ってよいか判断する。根拠は以下だけ。JSON だけ: "
                     '{"verdict": "入ってよい" か "慎重に" か "見送るべき", "reason": "40字以内"}\n\n'
                     f"【合図】{strategy} {'買い' if side == 1 else '売り'} 入る {entry:.3f} 損切り {stop:.3f} 利確 {target:.3f}\n"
                     f"理由: {' / '.join(reasons[:4])}\n注意: {' / '.join(cautions[:4]) or 'なし'}\n"
                     f"【全統括AI】{res.get('conclusion', '')} / 方向の会議 {res.get('verdict', '')} {res.get('overall', 0):+.2f} / "
                     f"リスク {res.get('risk', {}).get('level', '')}\n【24時間以内の重要指標】{ev}\n【直近6時間の見出し】\n{heads}")
        if not isinstance(r, dict):
            return ""
        v, why = str(r.get("verdict", ""))[:10], str(r.get("reason", ""))[:60]
        row = {"time_utc": f"{now:%Y-%m-%d %H:%M}", "strategy": strategy, "side": side, "entry": round(entry, 3),
               "stop": round(stop, 3), "target": round(target, 3), "verdict": v, "reason": why}
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        pd.DataFrame([row]).to_csv(PATH, mode="a", header=not os.path.exists(PATH), index=False)
        return f"統括補佐AI (Gemini・検証中): {v} — {why}"
    except Exception as e:
        print("統括補佐AIのエラー:", e)
        return ""


def score(h) -> list[str]:
    """週次: 補佐AIの答えごとに、その後の成績 (損切りと利確のどちらに先に届いたか、48時間まで) を比べる"""
    if not os.path.exists(PATH):
        return ["統括補佐AI: まだ記録なし"]
    d = pd.read_csv(PATH)
    out = []
    for v, g in d.groupby("verdict"):
        rs = []
        for r in g.itertuples():
            w = h[h.index > pd.Timestamp(r.time_utc, tz="UTC")].iloc[:48]
            res = None
            for hi, lo in zip(w.High, w.Low):
                if (r.side == 1 and lo <= r.stop) or (r.side == -1 and hi >= r.stop):
                    res = -1.0; break
                if (r.side == 1 and hi >= r.target) or (r.side == -1 and lo <= r.target):
                    res = abs(r.target - r.entry) / abs(r.entry - r.stop); break
            if res is None and len(w) >= 48:
                res = r.side * (w.Close.iat[-1] - r.entry) / abs(r.entry - r.stop)
            if res is not None:
                rs.append(res)
        if rs:
            out.append(f"統括補佐AI「{v}」: {len(rs)}件 平均 {sum(rs) / len(rs):+.2f}R")
    return out or ["統括補佐AI: 答え合わせできる記録なし"]
