"""情報部の「AIに読ませる」専門AI: Google の Gemini (無料枠) にニュースの見出しを読ませ、6つの専門の立場から意見を出させる。
①経済指標 ②金融政策 ③ニュース ④地政学 ⑤他市場 ⑥センチメント

費用を出さないための決まり:
- キー (GitHub の Secret `GEMINI_API_KEY`) が無ければ何もしない
- 支払い方法を登録していない Google のプロジェクトのキーを使う (無料枠を超えると失敗するだけで、請求は起きない)
- 呼ぶのは1時間に1回まで・1日 MAX_PER_DAY 回まで (state/gemini.json で数える)
- 送るのは公開ニュースの見出しと、ドル円の価格だけ (利用者の情報は送らない)
重みは記録を始めた日から検証部が採点して決める (それまでは0 = 説明にだけ使う)。"""
import json
import os
import time
import urllib.request

import pandas as pd

from .. import specialists as sp

STATE = os.getenv("FX_GEMINI_STATE", "state/gemini.json")
MAX_PER_DAY = 30
ROLES = ["経済指標", "金融政策", "ニュース", "地政学", "他市場", "センチメント"]
QUERIES = ["ドル円", "日銀 金融政策", "FRB 利下げ OR 利上げ", "米国 経済指標", "地政学 リスク", "米国債 利回り"]
API = "https://generativelanguage.googleapis.com/v1beta"


def _get(url, body=None, timeout=60):
    # キーは URL ではなくヘッダーで送る (エラー表示や記録にキーが残らないように)
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def _models() -> list[str]:
    """使える軽いモデル (flash) の候補を、優先する順に並べる。GEMINI_MODEL があればそれを先頭に"""
    ms = _get(f"{API}/models?pageSize=200").get("models", [])
    ok = [m["name"] for m in ms if "generateContent" in m.get("supportedGenerationMethods", [])]
    bad = ("image", "tts", "live", "audio", "embedding", "thinking", "native", "computer")
    flash = [m for m in ok if "flash" in m and not any(b in m for b in bad)]
    # 正式版 → lite → preview/exp の順。同じ中では新しい版を先に
    rank = lambda m: (("preview" in m or "exp" in m) * 2 + ("lite" in m), [-int(x) if x.isdigit() else 0 for x in m.replace("-", ".").split(".")])
    c = sorted(flash, key=rank)
    if os.getenv("GEMINI_MODEL"):
        c = [os.environ["GEMINI_MODEL"]] + [m for m in c if m != os.environ["GEMINI_MODEL"]]
    return c


def _load():
    try:
        return json.load(open(STATE))
    except Exception:
        return {}


def ask(price: float, now=None) -> dict | None:
    """最新の見出しを読ませて、専門ごとの意見を返す。呼べない時は前回の結果 (1時間以内) か None"""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    now = now or pd.Timestamp.now(tz="UTC")
    st = _load()
    day = f"{now:%Y-%m-%d}"
    if st.get("last") and time.time() - st.get("last_ts", 0) < 3300:
        return st.get("result")
    if st.get("day") == day and st.get("count", 0) >= MAX_PER_DAY:
        return st.get("result") if time.time() - st.get("last_ts", 0) < 3 * 3600 else None
    items, seen = [], set()
    for q in QUERIES:
        for x in sp.headlines(query=q, hours=24)[:15]:
            if x["title"] not in seen:
                seen.add(x["title"]); items.append(x)
    items = sorted(items, key=lambda x: x["time"], reverse=True)[:60]
    if not items:
        return None
    lines = "\n".join(f"- {(x['time'] + pd.Timedelta(hours=9)):%m/%d %H:%M} {x['title']}" for x in items)
    prompt = (
        f"あなたは為替 (米ドル/円) の調査チームです。今は日本時間 {(now + pd.Timedelta(hours=9)):%Y-%m-%d %H:%M}、"
        f"ドル円は {price:.2f} 円です。以下は直近24時間のニュースの見出しです。\n{lines}\n\n"
        "次の6人の専門家それぞれの立場で、今後24時間のドル円への影響を判断してください: "
        + "、".join(ROLES) + "。\n"
        "bias は -1 (ドル円が下がりやすい=円高) 〜 +1 (上がりやすい=円安ドル高)、材料がなければ0。"
        "importance は 0〜1 (相場を大きく動かしうる度合い)。reason は日本語で40字以内、根拠にした見出しの内容を含める。"
        "見出しに無いことは推測しない。JSON だけを返す: "
        '{"roles": [{"role": "経済指標", "bias": 0.0, "importance": 0.0, "reason": ""}, ...], "summary": "全体のまとめ60字以内"}')
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}}
    res, model, errs = None, None, []
    try:
        cands = ([st["model"]] if st.get("model") else []) + [m for m in _models() if m != st.get("model")]
    except Exception as e:
        print("Gemini のモデル一覧を取れませんでした:", str(e)[:120])
        cands = [st["model"]] if st.get("model") else []
    for m in cands[:4]:              # 無料枠が無い・使い切ったモデルは次の候補へ (1回の実行で4つまで)
        try:
            out = _get(f"{API}/{m}:generateContent", body)
            txt = out["candidates"][0]["content"]["parts"][0]["text"]
            res = json.loads(txt)
            model = m
            break
        except Exception as e:
            errs.append(f"{m.split('/')[-1]}: {str(e)[:60]}")
    if res is None:
        print("Gemini を呼べませんでした:", " / ".join(errs))
        st.update(last_ts=time.time() - 2400, model=None)    # 20分後にもう一度だけ試せるように
        os.makedirs(os.path.dirname(STATE) or ".", exist_ok=True)
        json.dump(st, open(STATE, "w"), ensure_ascii=False)
        return None
    res["model"] = model
    res["n_headlines"] = len(items)
    st = {"day": day, "count": (st.get("count", 0) if st.get("day") == day else 0) + 1, "last": f"{now:%Y-%m-%d %H:%M}",
          "last_ts": time.time(), "model": model, "result": res}
    os.makedirs(os.path.dirname(STATE) or ".", exist_ok=True)
    json.dump(st, open(STATE, "w"), ensure_ascii=False)
    return res


def opinions(price: float) -> list[dict]:
    res = ask(price)
    if not res:
        print("(AI情報) Gemini の結果なし" + ("" if os.getenv("GEMINI_API_KEY") else " (キー未設定)"))
        return []
    print(f"(AI情報) Gemini {res.get('model', '')} 見出し{res.get('n_headlines', 0)}本: {str(res.get('summary', ''))[:60]}")
    out = []
    for r in res.get("roles", []):
        if r.get("role") in ROLES:
            b = max(-1.0, min(1.0, float(r.get("bias", 0)))) * max(0.0, min(1.0, float(r.get("importance", 1))))
            out.append(dict(name=f"AI情報|{r['role']}", bias=b, reason=str(r.get("reason", ""))[:80]))
    if res.get("summary"):
        out.append(dict(name="AI情報|まとめ", bias=0.0, reason=str(res["summary"])[:100]))
    return out
