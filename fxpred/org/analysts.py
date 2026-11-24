"""専門AI。どれも「日本時間10時の時点で分かっている値」だけから、ドル円の向きの意見を出す。

意見 (bias) は -1 (売り寄り) 〜 +1 (買い寄り)。向きは「よく言われる理屈」で固定する
(例: 米金利が上がる → ドル買い)。過去のデータに合わせて向きを入れ替えることはしない
(入れ替えると、たまたまの偏りを拾ってしまうため)。理屈どおりに効かなかった専門AIは、検証部が重み0にする。

履歴で検証できない専門AI (ニュース・地政学) は、記録を始めた日から採点する。"""
import numpy as np
import pandas as pd

from .. import specialists as sp


def _z(s: pd.Series, n=250) -> pd.Series:
    return s / s.rolling(n, min_periods=60).std()


def _clip(s):
    return (s / 2).clip(-1, 1)


def _chg(x, col, n, log=True):
    v = x[col]
    return np.log(v).diff(n) if log else v.diff(n)


def features(x: pd.DataFrame, uj: pd.Series, h: pd.DataFrame | None = None) -> pd.DataFrame:
    """x: data.at_jst10 の表、uj: 日本時間10時のドル円。戻り値: 専門AIごとの意見 (列) × 日"""
    b = pd.DataFrame(index=x.index)
    # ---- 金融政策AI
    if "米2年" in x:
        b["金融政策|米2年金利の勢い(5日)"] = _clip(_z(_chg(x, "米2年", 5, False)))
        b["金融政策|米2年金利の流れ(20日)"] = _clip(_z(_chg(x, "米2年", 20, False)))
    if "FF金利" in x:
        b["金融政策|政策金利の方向(60日)"] = np.sign(_chg(x, "FF金利", 60, False).round(2)).fillna(0)
    if "米10年" in x and "日10年" in x:
        b["金融政策|日米金利差の変化(20日)"] = _clip(_z((x["米10年"] - x["日10年"]).diff(20)))
    # ---- 他市場AI
    b["他市場|米株の勢い(5日)"] = _clip(_z(_chg(x, "米株", 5)))
    b["他市場|日経の勢い(5日)"] = _clip(_z(_chg(x, "日経", 5)))
    b["他市場|金の勢い(5日)・逆"] = -_clip(_z(_chg(x, "金", 5)))
    b["他市場|原油の勢い(5日)"] = _clip(_z(_chg(x, "原油", 5)))
    b["他市場|ドル指数の勢い(5日)"] = _clip(_z(_chg(x, "ドル指数", 5)))
    b["他市場|米10年金利(前日)"] = _clip(_z(_chg(x, "米10年(Yahoo)", 1, False)))
    # ---- センチメントAI (リスク回避 = 円買い = ドル円の下げ)
    vix = x["VIX"]
    b["センチメント|VIXの高さ・逆"] = -_clip((vix - vix.rolling(250, min_periods=60).mean()) / vix.rolling(250, min_periods=60).std())
    b["センチメント|VIXの上がり方(5日)・逆"] = -_clip(_z(vix.diff(5)))
    # ---- ドル・円の強さAI (相場分析部)
    dxy = np.log(x["ドル指数"])
    usd_era = dxy - dxy.rolling(200, min_periods=100).mean()
    dip = _z(dxy.diff(5))
    b["強さ|ドル高の時代の押し目は戻る"] = np.where((usd_era > 0) & (dip < -1.5), 1.0, 0.0)
    lu = np.log(uj.reindex(x.index))
    yen = -(lu.diff(5) - dxy.diff(5))            # 円の強さ ≒ ドル円の動きのうち、ドル指数で説明できない分の逆
    b["強さ|円の勢い(5日)・逆"] = -_clip(_z(yen))
    b["強さ|ドルも強く円も弱い"] = np.where((_z(dxy.diff(5)) > 0.5) & (_z(yen) < -0.5), 1.0,
                                     np.where((_z(dxy.diff(5)) < -0.5) & (_z(yen) > 0.5), -1.0, 0.0))
    # ---- 相場分析部 (1時間足から)
    if h is not None:
        jst = h.index + pd.Timedelta(hours=9)
        c = h.Close
        e50, e200 = c.ewm(span=50).mean(), c.ewm(span=200).mean()
        hi, lo = h.High.rolling(120).max(), h.Low.rolling(120).min()
        pos = (c - lo) / (hi - lo)
        delta = c.diff()
        up, dn = delta.clip(lower=0).ewm(alpha=1 / 14).mean(), (-delta.clip(upper=0)).ewm(alpha=1 / 14).mean()
        rsi = 100 - 100 / (1 + up / dn)
        # 10時の判断に使えるのは 9時台の足まで (10時台の足はまだ確定していない)
        m = jst.hour == 9
        day = (h.index[m] + pd.Timedelta(hours=9)).normalize().tz_localize(None)
        tr = pd.Series(np.sign((e50 - e200)[m].values), index=day)
        rg = pd.Series(np.where((pos[m] <= .25) & (rsi[m] < 45), 1, np.where((pos[m] >= .75) & (rsi[m] > 55), -1, 0)), index=day)
        mo = pd.Series(np.sign(c.diff(24)[m].values), index=day)
        b["相場分析|トレンド(50/200本)"] = tr[~tr.index.duplicated()].reindex(x.index)
        b["相場分析|レンジの端で逆張り"] = rg[~rg.index.duplicated()].reindex(x.index)
        b["相場分析|前日からの勢い"] = mo[~mo.index.duplicated()].reindex(x.index)
    return b.astype(float)


# ---------------------------------------------------------------- 履歴で検証できない専門AI (今の値だけ)
GEO_WORDS = ["戦争", "紛争", "ミサイル", "攻撃", "侵攻", "制裁", "関税", "テロ", "地政学", "緊迫", "核"]
POLICY_WORDS = {"利上げ": 1, "タカ派": 1, "引き締め": 1, "利下げ": -1, "ハト派": -1, "緩和": -1}


def news_now(items: list[dict]) -> dict:
    """ニュースAI: 見出しの言葉から向きを数える (重要そうな見出し=速報・要人を2倍)"""
    s, used = 0.0, []
    for x in items[:40]:
        t = x["title"]
        w = 2.0 if any(k in t for k in ("速報", "総裁", "議長", "財務官", "財務相")) else 1.0
        v = w * (sum(k in t for k in sp.UP_WORDS) - sum(k in t for k in sp.DOWN_WORDS))
        if v:
            s += v; used.append(t)
    return {"bias": float(np.clip(s / 10, -1, 1)), "reason": used[0] if used else "目立つ見出しなし", "n": len(used)}


def geo_now(items: list[dict]) -> dict:
    """地政学AI: 紛争・制裁などの見出しが多いほどリスク回避 (円買い) とみる"""
    hits = [x["title"] for x in items[:60] if any(k in x["title"] for k in GEO_WORDS)]
    return {"bias": -min(len(hits) / 5, 1.0), "reason": hits[0] if hits else "目立つ地政学ニュースなし", "n": len(hits)}


def policy_news_now(items: list[dict]) -> dict:
    """金融政策AI (ニュース): 日銀・FRB の見出しから。FRB のタカ派 = ドル高 (+)、日銀のタカ派 = 円高 (-)"""
    s, used = 0.0, []
    for x in items[:60]:
        t = x["title"]
        side = 1 if any(k in t for k in ("FRB", "FOMC", "パウエル", "米連邦")) else -1 if any(k in t for k in ("日銀", "植田")) else 0
        if not side:
            continue
        v = sum(val for k, val in POLICY_WORDS.items() if k in t)
        if v:
            s += side * v; used.append(t)
    return {"bias": float(np.clip(s / 4, -1, 1)), "reason": used[0] if used else "政策の目立つ見出しなし", "n": len(used)}


def events_now(now_utc: pd.Timestamp) -> dict:
    """経済指標AI: 24時間以内の重要指標 (🔴) と、直前に終わった指標を伝える。向きは出さない (発表直後の動きは急変担当が扱う)"""
    try:
        from .. import calendar as cal
        ev = cal.events(impacts=("High",))
    except Exception:
        ev = []
    soon = [e for e in ev if pd.Timedelta(0) <= e["time"] - now_utc <= pd.Timedelta(hours=24)]
    names = [cal.fmt(e) for e in soon]
    return {"bias": 0.0, "reason": ("24時間以内の重要指標: " + " / ".join(names[:3])) if names else "24時間以内に重要指標なし",
            "n": len(soon), "caution": bool(soon)}
