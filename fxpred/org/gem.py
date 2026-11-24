"""Gemini (無料枠) を、ニュース以外の仕事にも使うための共通の呼び出しと「無料枠の残りの管理」。

無料枠の管理 (state/gemini_budget.json):
- 1日の上限 DAILY_CAP 回 (Google の無料枠の1日の区切りは米国太平洋時間の0時 ≒ 日本時間17時)。仕事ごとにも上限を置き、大事な仕事を優先する
- Google から「使いすぎ (429)」が返ったら、その日はもう呼ばない (翌日の区切りまで)。上限に近づいたら優先度の低い仕事から止める
- どのモデルが使えたかを覚えておく。支払い方法は登録していないので、無料枠を超えても失敗するだけで請求は起きない
送るのは公開ニュースの見出し・指標の名前・相場の数字・会議のまとめだけ (利用者の取引や個人の情報は送らない)。"""
import json
import os
import time

import pandas as pd

BUDGET = os.getenv("FX_GEMINI_BUDGET", "state/gemini_budget.json")
DAILY_CAP = int(os.getenv("GEMINI_DAILY_CAP", "200"))
# 仕事ごとの1日の上限と優先度 (小さいほど優先)。上限に近づくと優先度の低いものから止める
JOBS = {"急変の原因": (8, 0), "指標の結果": (16, 1), "ニュース": (30, 2), "朝の見立て": (2, 2), "反対意見": (16, 3), "週次の振り返り": (2, 3)}


def _day(now=None):
    now = now or pd.Timestamp.now(tz="UTC")
    return f"{(now.tz_convert('America/Los_Angeles')):%Y-%m-%d}"


def _load():
    try:
        b = json.load(open(BUDGET))
    except Exception:
        b = {}
    if b.get("day") != _day():
        b = {"day": _day(), "count": 0, "by": {}, "blocked": False, "model": b.get("model")}
    return b


def _save(b):
    os.makedirs(os.path.dirname(BUDGET) or ".", exist_ok=True)
    json.dump(b, open(BUDGET, "w"), ensure_ascii=False)


def allowed(job: str) -> bool:
    if not os.getenv("GEMINI_API_KEY"):
        return False
    b = _load()
    cap, pri = JOBS.get(job, (5, 3))
    if b.get("blocked") or b["by"].get(job, 0) >= cap:
        return False
    left = DAILY_CAP - b["count"]
    # 残りが2割を切ったら優先度2以上を、1割を切ったら優先度1以上を止める
    if (left < DAILY_CAP * 0.2 and pri >= 2) or (left < DAILY_CAP * 0.1 and pri >= 1) or left <= 0:
        return False
    return True


def note(job: str, ok=True, quota=False, model=None):
    b = _load()
    b["count"] += 1
    b["by"][job] = b["by"].get(job, 0) + 1
    if quota:
        b["blocked"] = True
    if ok and model:
        b["model"] = model
    _save(b)


def status() -> str:
    b = _load()
    return f"Gemini 今日 {b['count']}/{DAILY_CAP}回" + (" (無料枠の上限に達したため停止中)" if b.get("blocked") else "") + \
        " " + " ".join(f"{k}{v}" for k, v in b["by"].items())


def call(job: str, prompt: str, as_json=True, temperature=0.2) -> dict | str | None:
    """Gemini に1回たずねる。無料枠の管理を通し、呼べなければ None"""
    if not allowed(job):
        return None
    from . import llm
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, **({"responseMimeType": "application/json"} if as_json else {})}}
    b = _load()
    try:
        cands = ([b["model"]] if b.get("model") else []) + [m for m in llm._models() if m != b.get("model")]
    except Exception:
        cands = [b["model"]] if b.get("model") else []
    quota_hits = 0
    for m in cands[:3]:
        try:
            out = llm._get(f"{llm.API}/{m}:generateContent", body)
            txt = out["candidates"][0]["content"]["parts"][0]["text"]
            note(job, ok=True, model=m)
            return json.loads(txt) if as_json else txt.strip()
        except Exception as e:
            if "429" in str(e):
                quota_hits += 1
            continue
    note(job, ok=False, quota=quota_hits >= min(len(cands[:3]), 2) and quota_hits > 0)
    return None
