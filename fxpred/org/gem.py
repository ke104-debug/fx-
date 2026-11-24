"""Gemini (無料枠) を、ニュース以外の仕事にも使うための共通の呼び出しと「無料枠の残りの管理」。

無料枠の管理 (state/gemini_budget.json):
- 1日の上限 DAILY_CAP 回 (Google の無料枠の1日の区切りは米国太平洋時間の0時 ≒ 日本時間17時)。Google から上限の数字が返ってきたらそれを覚えて使う
- 普段の仕事は上限の NORMAL_SHARE (7割) まで。残り3割は緊急用 (急変の原因・重要指標の結果) にとっておき、普段の分を使い切っても緊急時だけは動ける
- 仕事ごとにも上限を置き、残りが減ったら優先度の低い仕事から止める
- 回数に加えて、送った文字の量 (トークン) も数える。Google から「使いすぎ (429)」が返ったら、その日はもう呼ばない
- どのモデルが使えたかを覚えておく。支払い方法は登録していないので、無料枠を超えても失敗するだけで請求は起きない
送るのは公開ニュースの見出し・指標の名前・相場の数字・会議のまとめだけ (利用者の取引や個人の情報は送らない)。"""
import json
import os
import time

import pandas as pd

BUDGET = os.getenv("FX_GEMINI_BUDGET", "state/gemini_budget.json")
DAILY_CAP = int(os.getenv("GEMINI_DAILY_CAP", "200"))   # Google の上限が分からない間の想定
NORMAL_SHARE = 0.7
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


def limit() -> int:
    """1日の上限 (Google が返した数字を覚えていればそれ、無ければ想定値)"""
    try:
        return int(json.load(open(BUDGET)).get("learned_rpd") or DAILY_CAP)
    except Exception:
        return DAILY_CAP


def allowed(job: str) -> bool:
    if not os.getenv("GEMINI_API_KEY"):
        return False
    b = _load()
    cap, pri = JOBS.get(job, (5, 3))
    if b.get("blocked"):
        # 上限に達した後も、急変の原因だけは1時間に1回は試す (別のモデルの枠が残っていることがある。失敗しても請求は起きない)
        if pri == 0 and time.time() - b.get("emergency_try", 0) > 3600:
            b["emergency_try"] = time.time()
            _save(b)
            return True
        return False
    if b["by"].get(job, 0) >= cap:
        return False
    lim = limit()
    normal = int(lim * NORMAL_SHARE)
    if pri == 0 or (pri == 1 and b["by"].get(job, 0) < 4):
        return b["count"] < lim - 2          # 緊急 (急変の原因・指標の結果の最初の数回) は予備の分まで使える
    left = normal - b["count"]
    # 普段の分の残りが3割を切ったら優先度3を、1割を切ったら優先度2以上を止める
    if left <= 0 or (left < normal * 0.3 and pri >= 3) or (left < normal * 0.1 and pri >= 2):
        return False
    return True


def note(job: str, ok=True, quota=False, model=None, tokens=0):
    b = _load()
    b["count"] += 1
    b["by"][job] = b["by"].get(job, 0) + 1
    b["tokens"] = b.get("tokens", 0) + int(tokens or 0)
    if not ok:
        b["errors"] = b.get("errors", 0) + 1
    if quota:
        b["blocked"] = True
    if ok and model:
        b["model"] = model
    _save(b)


def learn(err) -> bool:
    """429 の返事の中身から上限の数字を読み取って覚える。429 なら True"""
    try:
        import urllib.error
        if not isinstance(err, urllib.error.HTTPError) or err.code != 429:
            return False
        body = json.loads(err.read() or b"{}")
        b = _load()
        for d in body.get("error", {}).get("details", []):
            for v in d.get("violations", []):
                qid, val = str(v.get("quotaId", "")), v.get("quotaValue")
                if val and "PerDay" in qid:
                    b["learned_rpd"] = int(val)
                elif val and "PerMinute" in qid:
                    b["learned_rpm"] = int(val)
        b["last_429"] = f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M}"
        _save(b)
        return True
    except Exception:
        return True


def status() -> str:
    b = _load()
    lim = limit()
    normal = int(lim * NORMAL_SHARE)
    src = "Googleの上限" if b.get("learned_rpd") else "想定の上限"
    return (f"Gemini 今日 {b['count']}回 / 普段の分 {normal}・緊急の予備 {lim - normal} ({src} {lim}回)・"
            f"送った量 約{b.get('tokens', 0) // 1000}千トークン・失敗 {b.get('errors', 0)}回"
            + (" (無料枠の上限に達したため停止中)" if b.get("blocked") else ""))


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
            note(job, ok=True, model=m, tokens=out.get("usageMetadata", {}).get("totalTokenCount", 0))
            return json.loads(txt) if as_json else txt.strip()
        except Exception as e:
            if learn(e):
                quota_hits += 1
            continue
    note(job, ok=False, quota=quota_hits >= min(len(cands[:3]), 2) and quota_hits > 0)
    return None
