"""論文からの自主学習 (利用者の案 2026-11-27: 余った資源で論文などからシステム向上に役立つ内容を学ぶ)。
週1回 (週次の見直し)、arXiv の公開API (無料) で為替・トレンド・ニュースと価格などの新しい論文を集め、
AI (Groq/Gemini の無料枠) に『このシステムで23年のデータで確かめられる仮説』に要約させ、提案の一覧 (journal/proposals.csv) に登録する。
決まり: 論文の内容でルールを自動で変えない。仮説は必ず 作る/確かめる期間・偽物の相場・同じ流れの中で適当に入る場合 との比較を通してから影の運用へ。
読んだ論文は journal/papers.csv に残す (同じ論文は2度読まない)"""
import os
import re
import urllib.parse
import urllib.request

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
SEEN = os.path.join(ROOT, "journal", "papers.csv")
TOPICS = ['"foreign exchange" AND (trend OR momentum)', '"exchange rate" AND (news OR announcement) AND intraday',
          '"time series momentum" AND currency', '"carry trade" AND crash', '"central bank" AND intervention AND "exchange rate"',
          '"technical analysis" AND currency AND profitability']


def fetch(n_each=4) -> list[dict]:
    out = []
    for t in TOPICS:
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": f"all:{t}", "max_results": n_each, "sortBy": "submittedDate", "sortOrder": "descending"})
        try:
            xml = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "fxpred-research"}), timeout=30).read().decode()
        except Exception as e:
            print("arXiv を読めませんでした:", e)
            continue
        for ent in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
            g = lambda tag: (re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", ent, re.S) or [None, ""])[1].strip()
            out.append({"id": g("id"), "title": re.sub(r"\s+", " ", g("title")), "summary": re.sub(r"\s+", " ", g("summary"))[:1500],
                        "published": g("published")[:10]})
    return out


def run(max_new=6) -> list[str]:
    seen = pd.read_csv(SEEN) if os.path.exists(SEEN) else pd.DataFrame(columns=["id", "title", "read_utc", "hypothesis"])
    papers = [p for p in fetch() if p["id"] and p["id"] not in set(seen.id)]
    if not papers:
        return ["論文の自主学習: 新しい論文なし"]
    from .org import groq, proposals
    lines, rows = [], []
    for p in papers[:max_new]:
        r, who = groq.ask("検証の補助", "あなたは USD/JPY の売買システムの研究員。次の論文の要旨から、このシステム (1分足〜日足の23年分の価格、米金利、"
                          "指標の日時、投機筋の持ち高がある) で確かめられる仮説を1つだけ、具体的なルールとして書く。確かめられない (データが無い・別の市場だけ) なら空。"
                          '要旨に無い数字を作らない。JSON だけ: {"hypothesis": "80字以内 (空でもよい)", "test": "確かめ方 60字以内"}\n\n'
                          f"題名: {p['title']}\n要旨: {p['summary']}")
        hyp = str((r or {}).get("hypothesis", "")).strip() if isinstance(r, dict) else ""
        rows.append({"id": p["id"], "title": p["title"][:150], "read_utc": f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d}", "hypothesis": hyp[:120]})
        if hyp:
            proposals.add(f"論文の自主学習 ({who})", hyp, "", "", "", "提案", f"出典: {p['title'][:80]} {p['id']} / 確かめ方: {str(r.get('test', ''))[:60]}")
            lines.append(f"{hyp[:60]} (出典: {p['title'][:40]})")
    pd.concat([seen, pd.DataFrame(rows)]).to_csv(SEEN, index=False)
    return [f"論文の自主学習: 新しく {len(rows)}本を読み、確かめられる仮説 {len(lines)}件を提案に登録"] + lines[:4]
