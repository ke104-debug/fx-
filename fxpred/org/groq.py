"""Groq (無料枠) を補助のAIとして使う。ルールや重みは変えない (意見を記録・通知に添えるだけ)。

試用期間: UNTIL (2026-12-25) まで。過ぎたら自動で呼ばない (続けるかは利用者と相談)。
費用を出さない決まり:
- キー (GitHub の Secret `GROQ_API_KEY`) が無ければ何もしない。支払い方法を登録しない前提 (無料枠を超えたら失敗するだけ)
- Groq は返事のヘッダーに「今日の残りの回数・上限」を返すので、それを覚えて、残りが2割を切ったら呼ばない。仕事ごとにも1日の上限
- 「使いすぎ (429)」が返ったら、そのモデルはその日使わない
送るのは公開ニュースの見出し・会議のまとめ・合図の中身だけ (利用者の個人の情報や取引は送らない)。
仕事: 統括補佐AIのもう一人の意見 / 反対意見 / 全部隊の相談役 (毎時) / 監査の補助 / 検証部隊の補助 (週次)。Groq が使えない時は Gemini"""
import json
import os
import urllib.error
import urllib.request

import pandas as pd

API = "https://api.groq.com/openai/v1"
BUDGET = os.getenv("FX_GROQ_BUDGET", "state/groq_budget.json")
UNTIL = pd.Timestamp("2026-12-25", tz="UTC")
JOBS = {"統括補佐": 12, "反対意見": 16, "相談役": 26, "監査の補助": 4, "検証の補助": 2, "調査": 320}
DAY_CAP = 500   # 利用者の指示: 1日1,000回の無料枠のうち500回までは使ってよい


def _day():
    return f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d}"


def _load():
    try:
        b = json.load(open(BUDGET))
    except Exception:
        b = {}
    if b.get("day") != _day():
        b = {"day": _day(), "count": 0, "by": {}, "errors": 0, "exhausted": [], "model": b.get("model"), "limits": {}}
    return b


def _save(b):
    os.makedirs(os.path.dirname(BUDGET) or ".", exist_ok=True)
    json.dump(b, open(BUDGET, "w"), ensure_ascii=False)


def _req(path, body=None, timeout=60):
    req = urllib.request.Request(f"{API}/{path}", data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
                                          "User-Agent": "fxpred"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), dict(r.headers)


def models() -> list[str]:
    """文章を作れるモデルを、大きい (賢い) 順に。GROQ_MODEL があれば先頭に"""
    d, _ = _req("models")
    ids = [m["id"] for m in d.get("data", []) if m.get("active", True)]
    bad = ("whisper", "tts", "guard", "playai", "embed", "vision", "distil", "orpheus", "prompt")
    ids = [m for m in ids if not any(b in m.lower() for b in bad)]
    size = lambda m: max([int(x[:-1]) for x in m.lower().replace("-", " ").replace("_", " ").replace("/", " ").split()
                          if x.endswith("b") and x[:-1].isdigit()] or [0])
    ids.sort(key=lambda m: -size(m))
    if os.getenv("GROQ_MODEL"):
        ids = [os.environ["GROQ_MODEL"]] + [m for m in ids if m != os.environ["GROQ_MODEL"]]
    return ids


def allowed(job: str) -> bool:
    if not os.getenv("GROQ_API_KEY") or pd.Timestamp.now(tz="UTC") >= UNTIL:
        return False
    b = _load()
    if b["by"].get(job, 0) >= JOBS.get(job, 4):
        return False
    if b["count"] >= DAY_CAP and job != "統括補佐":     # 合図の最終確認だけは上限の後も (残り2割の手前まで)
        return False
    lim = b.get("limits", {}).get(b.get("model") or "", {})
    if lim.get("limit") and lim.get("remaining") is not None and lim["remaining"] < 0.2 * lim["limit"]:
        return False
    return True


def status() -> str:
    b = _load()
    lim = b.get("limits", {}).get(b.get("model") or "", {})
    left = f"残り {lim.get('remaining')}/{lim.get('limit')}回" if lim.get("limit") else "残り不明"
    return (f"Groq 今日 {b['count']}回 ({(b.get('model') or '-').split('/')[-1]} {left}・失敗 {b.get('errors', 0)}回"
            + f"・試用は {UNTIL:%m/%d} まで)")


def call(job: str, prompt: str, as_json=True) -> dict | str | None:
    if not allowed(job):
        return None
    b = _load()
    try:
        ms = models()
    except Exception as e:
        print("Groq のモデル一覧を取れませんでした:", str(e)[:100])
        ms = [b["model"]] if b.get("model") else []
    ms = [m for m in ([b["model"]] if b.get("model") in ms else []) + ms if m not in b.get("exhausted", [])]
    errs = []
    for m in list(dict.fromkeys(ms))[:3]:
        body = {"model": m, "temperature": 0.2, "messages": [{"role": "user", "content": prompt}]}
        if as_json:
            body["response_format"] = {"type": "json_object"}
        try:
            out, hd = _req("chat/completions", body)
            txt = out["choices"][0]["message"]["content"]
            b = _load()
            b["count"] += 1
            b["by"][job] = b["by"].get(job, 0) + 1
            b["model"] = m
            try:   # 返事のヘッダーから、今日の残りの回数と上限を覚える
                b.setdefault("limits", {})[m] = {"limit": int(hd.get("x-ratelimit-limit-requests") or hd.get("X-Ratelimit-Limit-Requests")),
                                                 "remaining": int(hd.get("x-ratelimit-remaining-requests") or hd.get("X-Ratelimit-Remaining-Requests"))}
            except Exception:
                pass
            _save(b)
            return json.loads(txt) if as_json else txt.strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                b = _load()
                b.setdefault("exhausted", []).append(m)
                _save(b)
            errs.append(f"{m}: {e.code}")
        except Exception as e:
            errs.append(f"{m}: {str(e)[:60]}")
    b = _load()
    b["errors"] = b.get("errors", 0) + 1
    _save(b)
    print(f"Groq ({job}) を呼べませんでした:", " / ".join(errs) or "モデルなし")
    return None


def ask(job: str, prompt: str, as_json=True):
    """Groq を先に使い、だめなら Gemini (無料枠の分担)"""
    r = call(job, prompt, as_json)
    if r is not None:
        return r, "Groq"
    from . import gem
    return gem.call(job, prompt, as_json), "Gemini"
