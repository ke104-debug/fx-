"""調査班 (Groq 無料枠): ほかの担当 (相談役) が「調べてほしい」と頼んだことと、毎時の定番の調べものを、
Google ニュースの見出しから集めて Groq に読ませ、要点・ドル円への影響・確かさをまとめる。
Groq は自分でネットを見ないので、集めるのはこのプログラム (無料の RSS)。結果は journal/research.csv と、次の会議の中身に入る。
回数: 30分ごと: 定番 5件 (1時間に1回) + 依頼 最大6件 (1日 約400回まで)。Groq の1日の合計は 800回まで (groq.DAY_CAP)。判断の重みは0 (記録して採点)"""
import json
import os

import pandas as pd


def _append(path, df):
    """記録に行を足す。列が増えた時は、古い行と合わせて書き直す (列がずれて壊れないように)"""
    if os.path.exists(path):
        old = pd.read_csv(path)
        if list(old.columns) != list(df.columns):
            pd.concat([old, df], ignore_index=True).to_csv(path, index=False)
            return
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
PATH = os.path.join(ROOT, "journal", "research.csv")
QUEUE = os.getenv("FX_RESEARCH_QUEUE", "state/research_queue.json")
STANDING = ["ドル円 介入 OR 財務官", "FRB 高官 発言", "日銀 総裁 OR 審議委員 発言", "米国債 利回り", "円相場 要因"]
PER_HOUR = 6        # 30分ごとの依頼の上限 (定番は1時間に1回)


def request(unit: str, queries: list[str]):
    """相談役などからの調査の依頼を順番待ちに入れる (同じ言葉は1回だけ)"""
    try:
        q = json.load(open(QUEUE))
    except Exception:
        q = []
    have = {x["q"] for x in q}
    for s in queries[:2]:
        s = str(s).strip()[:40]
        # 相談役の言葉があいまいだと暗号資産などの関係ない記事を拾うので、為替の言葉が無ければ「ドル円」を付ける
        if s and not any(k in s for k in ("ドル", "円", "FRB", "日銀", "米", "為替", "金利", "介入")):
            s = "ドル円 " + s
        if s and s not in have:
            q.append({"unit": unit, "q": s})
            have.add(s)
    os.makedirs(os.path.dirname(QUEUE) or ".", exist_ok=True)
    json.dump(q[-40:], open(QUEUE, "w"), ensure_ascii=False)


def run(now=None) -> list[dict]:
    from . import groq
    from .. import specialists as sp
    now = now or pd.Timestamp.now(tz="UTC")
    mark = QUEUE + ".last"
    slot = f"{now.floor('30min'):%Y-%m-%d %H:%M}"
    try:   # 30分に1回だけ (毎時の分析は1時間に2回動く)
        if open(mark).read().strip() == slot:
            return []
    except Exception:
        pass
    os.makedirs(os.path.dirname(mark) or ".", exist_ok=True)
    open(mark, "w").write(slot)
    try:
        q = json.load(open(QUEUE))
    except Exception:
        q = []
    todo = ([{"unit": "定番", "q": s} for s in STANDING] if now.minute < 30 else []) + q[:PER_HOUR]
    rest = q[PER_HOUR:]
    rows, heads_by = [], {}
    for t in todo:
        if not groq.allowed("調査"):
            break
        import time as _t
        _t.sleep(2)     # 1分あたりの量の上限に当たらないよう少し間をあける
        heads = sp.headlines(query=t["q"], hours=24)[:15]
        if not heads:
            continue
        lines = "\n".join(f"- {(x['time'] + pd.Timedelta(hours=9)):%m/%d %H:%M} {x['title']}" for x in heads)
        heads_by[t["q"]] = lines[:1500]
        r = groq.call("調査", f"FX (ドル円) の調査班。「{t['q']}」について直近24時間のニュースの見出し:\n{lines}\n"
                      "見出しに書かれていることだけを使い、推測はしない。JSON だけ: "
                      '{"要点": "60字以内", "ドル円への影響": -1〜1 (円安ドル高=+), "確かさ": 0〜1, "新しさ": "最も新しい見出しの時刻 MM/DD HH:MM"}')
        if not isinstance(r, dict):
            continue
        try:
            eff = max(-1.0, min(1.0, float(r.get("ドル円への影響", 0))))
            conf = max(0.0, min(1.0, float(r.get("確かさ", 0))))
        except Exception:
            eff, conf = 0.0, 0.0
        rows.append({"time_utc": f"{now:%Y-%m-%d %H:%M}", "unit": t["unit"], "query": t["q"], "summary": str(r.get("要点", ""))[:90],
                     "effect": round(eff, 2), "confidence": round(conf, 2), "newest": str(r.get("新しさ", ""))[:12], "n": len(heads)})
    if rows and now.minute < 30:   # Gemini が Groq のまとめを見出しと照らして確かめる (1時間に1回、最初の3件)
        try:
            from . import gem
            chk = gem.call("調査の確認", "FX の調査班 (別のAI) が見出しからまとめた要点が、見出しの内容と合っているかを確かめる。"
                           "見出しに無いことを書いていたら「疑わしい」。JSON だけ: {\"checks\": [\"合っている\" か \"疑わしい\", ...]} (順番どおり)\n\n"
                           + "\n\n".join(f"{i + 1}. 調べたこと: {x['query']}\n要点: {x['summary']}\n見出し:\n{heads_by.get(x['query'], '')}"
                                           for i, x in enumerate(rows[:3])))
            if isinstance(chk, dict):
                for x, v in zip(rows[:3], chk.get("checks", [])):
                    x["checked"] = str(v)[:6]
        except Exception as e:
            print("調査の確認のエラー:", e)
    if rows:
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        _append(PATH, pd.DataFrame(rows))
    try:
        json.dump(rest, open(QUEUE, "w"), ensure_ascii=False)
    except Exception:
        pass
    return rows


def recent(hours=6, n=6) -> list[str]:
    """会議の中身に入れる、直近の調査の要点"""
    if not os.path.exists(PATH):
        return []
    d = pd.read_csv(PATH)
    d = d[pd.to_datetime(d.time_utc, utc=True) >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)]
    d = d.sort_values("confidence", ascending=False).drop_duplicates("query").head(n)
    return [f"{r.query}: {r.summary} (影響 {r.effect:+.1f}・確かさ {r.confidence:.1f})" for r in d.itertuples()]
