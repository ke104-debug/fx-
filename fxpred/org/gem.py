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
JOBS = {"急変の原因": (8, 0), "指標の結果": (16, 1), "ニュース": (30, 2), "朝の見立て": (2, 2), "監査の補助": (4, 2),
        "相談役": (10, 3), "反対意見": (16, 3), "週次の振り返り": (2, 3)}


def _day(now=None):
    now = now or pd.Timestamp.now(tz="UTC")
    return f"{(now.tz_convert('America/Los_Angeles')):%Y-%m-%d}"


def _load():
    try:
        b = json.load(open(BUDGET))
    except Exception:
        b = {}
    if b.get("day") != _day():
        b = {"day": _day(), "count": 0, "by": {}, "blocked": False, "model": b.get("model"),
             "learned_rpd": b.get("learned_rpd"), "n_models": b.get("n_models", 1), "exhausted": []}
    return b


def _save(b):
    os.makedirs(os.path.dirname(BUDGET) or ".", exist_ok=True)
    json.dump(b, open(BUDGET, "w"), ensure_ascii=False)


def limit() -> int:
    """1日の上限。Google の無料枠はモデルごとに1日の回数が決まっている (2026-11 時点で1モデル20回) ので、
    覚えた1モデルの上限 × 使えるモデルの数。分からない間は想定値"""
    try:
        b = json.load(open(BUDGET))
    except Exception:
        b = {}
    if b.get("learned_rpd"):
        if b.get("models_day") != _day() and os.getenv("GEMINI_API_KEY"):
            try:   # 使えるモデルの数を1日1回数える (モデル一覧の取得は無料枠の回数に入らない)
                from . import llm
                b = _load()
                b["n_models"], b["models_day"] = len(llm._models()), _day()
                _save(b)
            except Exception:
                pass
        return int(b["learned_rpd"]) * max(1, int(b.get("n_models", 1)))
    return DAILY_CAP


def usable(cands: list[str]) -> list[str]:
    """今日の枠を使い切ったモデルを外す"""
    ex = set(_load().get("exhausted", []))
    return [m for m in cands if m not in ex]


def exhausted(model: str):
    b = _load()
    b.setdefault("exhausted", [])
    if model and model not in b["exhausted"]:
        b["exhausted"].append(model)
    _save(b)


def set_models(n: int):
    b = _load()
    b["n_models"] = n
    _save(b)


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


def learn(err) -> str:
    """429 の返事の中身から上限の数字を読み取って覚える。
    戻り値: "day" (今日の枠を使い切った) / "minute" (1分あたりの上限。少し待てば使える) / "429" (種類不明) / "" (429 以外)"""
    try:
        import urllib.error
        if not isinstance(err, urllib.error.HTTPError) or err.code != 429:
            return ""
        body = json.loads(err.read() or b"{}")
        b = _load()
        kind = "429"
        for d in body.get("error", {}).get("details", []):
            for v in d.get("violations", []):
                qid, val = str(v.get("quotaId", "")), v.get("quotaValue")
                if "PerDay" in qid:
                    kind = "day"
                    if val:
                        b["learned_rpd"] = int(val)
                elif "PerMinute" in qid:
                    kind = "minute" if kind != "day" else kind
                    if val:
                        b["learned_rpm"] = int(val)
        b["last_429"] = f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} {kind}"
        _save(b)
        return kind
    except Exception:
        return "429"


def status() -> str:
    b = _load()
    lim = limit()
    normal = int(lim * NORMAL_SHARE)
    src = "Googleの上限" if b.get("learned_rpd") else "想定の上限"
    return (f"Gemini 今日 {b['count']}回 / 普段の分 {normal}・緊急の予備 {lim - normal} ({src} {lim}回"
            + (f" = 1モデル{b['learned_rpd']}回×{b.get('n_models', 1)}モデル" if b.get("learned_rpd") else "")
            + f"、使い切ったモデル {len(b.get('exhausted', []))})・"
            f"送った量 約{b.get('tokens', 0) // 1000}千トークン・失敗 {b.get('errors', 0)}回"
            + (" (無料枠の上限に達したため停止中)" if b.get("blocked") else ""))


def call(job: str, prompt: str, as_json=True, temperature=0.2) -> dict | str | None:
    """Gemini に1回たずねる。無料枠の管理を通し、呼べなければ None"""
    if not allowed(job):
        print(f"Gemini ({job}) は無料枠の管理で見送り: {status()}")
        return None
    from . import llm
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, **({"responseMimeType": "application/json"} if as_json else {})}}
    b = _load()
    try:
        allm = llm._models()
        set_models(len(allm))
        cands = ([b["model"]] if b.get("model") in allm else []) + [m for m in allm if m != b.get("model")]
    except Exception:
        cands = [b["model"]] if b.get("model") else []
    cands = usable(cands)
    if not cands:
        note(job, ok=False, quota=True)
        return None
    quota_hits, errs = 0, []
    for m in cands[:3]:
        try:
            out = llm._get(f"{llm.API}/{m}:generateContent", body)
            txt = out["candidates"][0]["content"]["parts"][0]["text"]
            note(job, ok=True, model=m, tokens=out.get("usageMetadata", {}).get("totalTokenCount", 0))
            return json.loads(txt) if as_json else txt.strip()
        except Exception as e:
            kind = learn(e)
            if kind:
                quota_hits += 1
            if kind == "day":
                exhausted(m)          # このモデルは今日の枠を使い切った → 次のモデルへ (1分の上限なら次の回に回す)
            errs.append(f"{m.split('/')[-1]}: {str(e)[:80]}")
            continue
    print(f"Gemini ({job}) を呼べませんでした:", " / ".join(errs) or "候補のモデルなし")
    note(job, ok=False, quota=not usable(cands))     # 全部のモデルが使い切った時だけ、その日は止める
    return None
