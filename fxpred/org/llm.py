"""情報部の「AIに読ませる」専門AI: Google の Gemini (無料枠) にニュースの見出しを読ませ、6つの専門の立場から意見を出させる。
①経済指標 ②金融政策 ③ニュース ④地政学 ⑤他市場 ⑥センチメント

費用を出さないための決まり:
- キー (GitHub の Secret `GEMINI_API_KEY`) が無ければ何もしない
- 支払い方法を登録していない Google のプロジェクトのキーを使う (無料枠を超えると失敗するだけで、請求は起きない)
- 呼ぶのは1時間に1回まで・1日 MAX_PER_DAY 回まで (state/gemini.json)。ほかの仕事と合わせた無料枠の管理は gem.py
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
ROLES = ["経済指標", "金融政策", "ニュース", "地政学", "他市場", "センチメント", "日本の政治・財政", "ドルか円か"]
QUERIES = ["ドル円", "日銀 金融政策", "FRB 利下げ OR 利上げ", "米国 経済指標", "地政学 リスク", "米国債 利回り",
           "為替介入 財務官 OR 財務相", "首相 財政 OR 選挙 OR 補正予算"]
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
    jh = (now + pd.Timedelta(hours=9)).hour
    every = 3300 if (jh >= 10 or jh < 2) else 3 * 3600 - 300      # 無料枠の節約: 活動時間外 (2〜10時) は3時間に1回
    if st.get("last") and time.time() - st.get("last_ts", 0) < every:
        return st.get("result")
    from . import gem
    if not gem.allowed("ニュース"):          # 無料枠の管理 (gem.py) で止められている
        return st.get("result") if time.time() - st.get("last_ts", 0) < 3 * 3600 else None
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
        "「ドルか円か」は、最近の動きがドル側の理由 (米国の材料) か円側の理由 (日本の材料) かを reason に書き、bias はその材料の向き。"
        "「日本の政治・財政」は選挙・首相交代・財政拡大・財務大臣や財務官の発言を見る。"
        "加えて intervention_stage に、財務大臣・財務官の円安けん制の強さを 0〜4 の整数で入れる "
        "(0=発言なし, 1=注視している, 2=憂慮・行き過ぎ, 3=断固たる措置・あらゆる手段を排除しない, 4=介入の実施や確認)。"
        "news_time に、判断の根拠にした最も新しい見出しの時刻 (日本時間 MM/DD HH:MM) を入れる。"
        "見出しに無いことは推測しない。JSON だけを返す: "
        '{"roles": [{"role": "経済指標", "bias": 0.0, "importance": 0.0, "reason": ""}, ...], "summary": "全体のまとめ60字以内", '
        '"intervention_stage": 0, "news_time": "MM/DD HH:MM"}')
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}}
    res, model, errs, tokens = None, None, [], 0
    try:
        allm = _models()
        gem.set_models(len(allm))
        cands = ([st["model"]] if st.get("model") in allm else []) + [m for m in allm if m != st.get("model")]
    except Exception as e:
        print("Gemini のモデル一覧を取れませんでした:", str(e)[:120])
        cands = [st["model"]] if st.get("model") else []
    cands = gem.usable(cands)
    for m in cands[:4]:              # 無料枠が無い・使い切ったモデルは次の候補へ (1回の実行で4つまで)
        try:
            out = _get(f"{API}/{m}:generateContent", body)
            txt = out["candidates"][0]["content"]["parts"][0]["text"]
            res = json.loads(txt)
            model = m
            tokens = out.get("usageMetadata", {}).get("totalTokenCount", 0)
            break
        except Exception as e:
            q = gem.learn(e)
            if q:
                gem.exhausted(m)
            errs.append(f"{m.split('/')[-1]}: {'429 ' if q else ''}{str(e)[:60]}")
    gem.note("ニュース", ok=res is not None, model=model, tokens=tokens,
             quota=res is None and not gem.usable(cands))
    if res is None:
        print("Gemini を呼べませんでした:", " / ".join(errs))
        st.update(last_ts=time.time() - 2400, model=None)    # 20分後にもう一度だけ試せるように
        os.makedirs(os.path.dirname(STATE) or ".", exist_ok=True)
        json.dump(st, open(STATE, "w"), ensure_ascii=False)
        return None
    res["model"] = model
    res["n_headlines"] = len(items)
    res["asked_at"] = f"{now:%Y-%m-%d %H:%M}"
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
    stg = res.get("intervention_stage")
    if isinstance(stg, (int, float)):
        out.append(dict(name="AI情報|介入けん制の段階", bias=0.0, stage=int(stg),
                        reason=["発言なし", "注視", "憂慮・行き過ぎ", "断固たる措置・あらゆる手段", "介入の実施・確認"][max(0, min(4, int(stg)))]))
    # 情報の鮮度: いつの情報にもとづく結論か (古い結果を使い回していないか)
    for o in out:
        o["as_of"] = res.get("asked_at", "")
    return out
