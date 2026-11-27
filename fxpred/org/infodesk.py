"""情報部 (統廃合後, 2026-11-27): 情報解析部隊と相場分析部隊をまとめたもの。票は持たず、報告に「環境メモ」を書く。
向き (買い寄り・売り寄り) は出さない。23年で単独では向きを当てられなかったため。担当の採点は会議の記録で裏で続ける。"""
import pandas as pd

JST = pd.Timedelta(hours=9)


def memo(h: pd.DataFrame, res: dict | None = None) -> list[str]:
    out = []
    if res:
        rk = res.get("risk", {})
        out.append(f"リスク管理部: {rk.get('level', '-')}" + "".join(f" / {w}: {y}" for w, y in rk.get("items", [])[:2]))
    try:   # 介入の段階
        from . import analysts
        from .. import specialists as sp
        iv = analysts.intervention_now(sp.headlines(query="為替介入 OR 財務官 OR 財務相", hours=48) + sp.headlines(hours=24))
        out.append(f"介入けん制の段階: {iv['stage']}/4" + (f" ({iv['reason'][:40]})" if iv.get("reason") else ""))
    except Exception as e:
        print("情報部 (介入) のエラー:", e)
    try:   # 24時間以内の重要指標
        from .. import calendar as cal
        now = pd.Timestamp.now(tz="UTC")
        ev = [e for e in cal.events(impacts=("High",)) if now <= e["time"] <= now + pd.Timedelta(hours=24)]
        out.append("24時間以内の重要指標: " + (" / ".join(f"{(e['time'] + JST):%m/%d %H:%M} {e['title']}" for e in ev[:4]) or "なし"))
    except Exception as e:
        print("情報部 (指標) のエラー:", e)
    try:   # 調査班の要点
        from . import research
        rr = research.recent(hours=12, n=8)
        seen, keep = set(), []
        for x in rr:      # ドル円と関係のない話題 (暗号資産など) と、同じ話の重複を外す
            if any(w in x for w in ("暗号資産", "仮想通貨", "ビットコイン", "レバレッジ1000")):
                continue
            k = x.split(":", 1)[-1][:25]
            if k not in seen:
                seen.add(k); keep.append(x)
        out += [f"ニュースの要点: {x[:70]}" for x in keep[:3]]
    except Exception as e:
        print("情報部 (調査) のエラー:", e)
    try:   # 米金利の動き (会議で唯一、両期間で効いた材料。向きの判断には使わず事実だけ)
        ops = (res or {}).get("ops", [])
        us = [o for o in ops if o["name"] == "金融政策|米2年金利の勢い(5日)"]
        if us and us[0].get("reason"):
            out.append(f"米2年金利 (5日): {us[0]['reason'][:50]}")
    except Exception:
        pass
    return out
