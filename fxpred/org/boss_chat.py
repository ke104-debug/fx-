"""ボスAIとの会話 (利用者の要望 2026-11-27: Claude Code からボスAIに話しかけたい)。
質問に、記録 (未解決の問題・ボスの判断・点検の結果・合図・作戦の成績・過去のミス・競争リーグ) だけを根拠に答える。
記録に無いことは『記録に無い』と答える。会話でルールは変えない (指示や指摘は台帳に残すだけ)。
使い方: GitHub Actions『ボスAIに話しかける』(boss-chat.yml) / 手元からは tools/ask_boss.sh "質問"。記録 journal/boss_chat.csv (公開されるので金額・持ち高は書かない)"""
import json
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
J = lambda *p: os.path.join(ROOT, *p)
LOG = J("journal", "boss_chat.csv")


def _tail(name, n, cols=None):
    try:
        d = pd.read_csv(J("journal", name), dtype=str)
        d = d[cols] if cols else d
        return d.tail(n).to_csv(index=False)[:2500]
    except Exception as e:
        return f"(読めない: {str(e)[:40]})"


def context() -> str:
    from . import boss
    parts = ["## 未解決の問題 (台帳)", "\n".join(boss.open_issues()) or "なし",
             "## ボスの最近の判断 (boss.csv)", _tail("boss.csv", 8),
             "## 点検部隊の NG (最近)", _tail("inspector.csv", 400).splitlines()[0] + "\n" + "\n".join(
                 [l for l in _tail("inspector.csv", 400).splitlines() if ",False," in l][-12:]),
             "## 最近の合図 (signals.csv)", _tail("signals.csv", 10),
             "## 決済した取引 (closed.csv)", _tail("closed.csv", 10),
             "## 過去の会話", _tail("boss_chat.csv", 4)]
    for name, f in [("通知の設定 (live.json)", "live.json"), ("作戦の成績 (23年)", "strategy_stats.json")]:
        try:
            parts += [f"## {name}", json.dumps(json.load(open(J("config", f))), ensure_ascii=False)[:2000]]
        except Exception:
            pass
    try:
        m = json.load(open(J("config", "known_mistakes.json")))
        parts += ["## 過去のミス (最近5件)", "\n".join(f"{x['id']}: {x['what'][:150]}" for x in m[-5:])]
    except Exception:
        pass
    try:
        from .. import league
        parts += ["## 組織の競争リーグ", "\n".join(league.table())]
    except Exception:
        pass
    return "\n".join(parts)


def ask(question: str, asker: str = "利用者") -> str:
    from . import groq
    prompt = ("あなたは USD/JPY 通知システムを率いる『ボスAI』。役目は各部署の暴走と甘やかし合いを引き締めること。"
              "次の記録だけを根拠に、日本語で短く (400字以内) 率直に答える。記録に無いことは『記録に無い』と言い、推測で数字を作らない。"
              "問題があれば隠さず言う。会話でルールは変えられない (変更は検証と利用者の承認が必要) ことを守る。\n\n"
              f"# 記録\n{context()}\n\n# 質問 ({asker})\n{question}")
    r, who = groq.ask("ボスとの会話", prompt, as_json=False)
    ans = (r if isinstance(r, str) else "") .strip() or "(AI を呼べず答えられませんでした)"
    row = {"time_utc": f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M}", "asker": asker, "question": question[:300],
           "answer": ans[:1500], "ai": who}
    old = pd.read_csv(LOG, dtype=str) if os.path.exists(LOG) else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])]).to_csv(LOG, index=False)
    return f"ボスAI ({who}): {ans}"


if __name__ == "__main__":
    import sys
    print(ask(os.getenv("Q") or " ".join(sys.argv[1:]), os.getenv("ASKER", "利用者")))
