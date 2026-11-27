"""流れ × タイミング (利用者の手法 2026-11-27): 大きな流れを分析 → 何度も見て流れの変化をとらえる → 5分監視で細かい流れ → テクニカルで入る時を探す。
研究と通知で同じ関数を使う (検証と通知の食い違いを防ぐ)。

流れ (15分足の本数で直接計算):
  大きな流れ = 3日の指数平均の向き / 中くらいの流れ = 半日の指数平均の向き / 細かい流れ = 15分足の平均線 (20>50)
  (2026-11-27 M015: 以前は explore.bars の up_d・up_4h を使っていたが、1日の本数の計算の誤りで実際は約3日と半日の流れだった。
   流れの長さを変えた比較で 2〜5日 + 0.5〜1日 が安定して良く、20日では弱い (+0.03〜0.08R) ので、3日 + 半日を明示して使う)
入る時 (トレンド転換ライン): 大きな流れと中くらいの流れがそろった向きの中の押し目 (戻り) で、切り下がる (切り上がる) 2つの高値 (安値) を結んだ線を
  15分足の終値で抜けた足 → 次の足の始値で入る。損切り 1ATR (15分足)・1R 進んだら建値・その後は最も伸びた値から 2ATR 後ろ・最長24時間。日本時間10時〜翌1時。
23年 (1分足から作った15分足・往復0.2pips): 年247回・+0.113R・作る +0.078 / 確かめ +0.153 / 2022年以降 +0.235R・プラスの年78%。
同じ流れの中で適当に入るより良い (200回の比較すべて)。弱点: 15分遅れで +0.096R、30分で +0.063R。手数料0.8pipsではほぼ0"""
import numpy as np
import pandas as pd

W, MAXD = 3, 30


def pivots(H, L, w=W):
    n = len(H); ph = np.zeros(n, bool); pl = np.zeros(n, bool)
    for i in range(w, n - w):
        ph[i] = H[i] == H[i - w:i + w + 1].max()
        pl[i] = L[i] == L[i - w:i + w + 1].min()
    return ph, pl


BIG_DAYS, MID_DAYS, PER_DAY = 3.0, 0.5, 96     # 15分足。流れの長さは 23年の比較で 2〜5日 + 0.5〜1日 が安定して良い (中心を採用)


def _slope(c: pd.Series, days: float) -> np.ndarray:
    n = int(days * PER_DAY)
    return (c.ewm(span=n).mean().diff(max(n // 20, 1)) > 0).values


def flows(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(大きな流れ = 3日の指数平均の向き, 中くらいの流れ = 半日の指数平均の向き)。15分足の本数で直接決める (足の作り方に左右されない)"""
    return _slope(d.Close, BIG_DAYS), _slope(d.Close, MID_DAYS)


def trend(d: pd.DataFrame) -> np.ndarray:
    """+1 = 大きな流れと中くらいの流れがそろって上、-1 = そろって下、0 = そろわない"""
    up, up4 = flows(d)
    return np.where(up & up4, 1, np.where((~up) & (~up4), -1, 0))


def trendline_signal(d: pd.DataFrame, w=W, maxd=MAXD) -> np.ndarray:
    """各足の合図 (+1 買い / -1 売り / 0)。スイングは w 本後に確定したものだけを使う (未来を見ない)"""
    c, H, L = d.Close.values, d.High.values, d.Low.values
    tr = trend(d)
    ph, pl = pivots(H, L, w)
    out = np.zeros(len(d), int); his, los = [], []
    for i in range(len(d)):
        k = i - w
        if k >= 0 and ph[k]:
            his.append((k, H[k]))
        if k >= 0 and pl[k]:
            los.append((k, L[k]))
        if i == 0:
            continue
        if tr[i] == 1 and len(his) >= 2:
            (k1, h1), (k2, h2) = his[-2], his[-1]
            if h2 < h1 and i - k2 <= maxd:
                s = (h2 - h1) / (k2 - k1)
                if c[i] > h2 + s * (i - k2) and c[i - 1] <= h2 + s * (i - 1 - k2):
                    out[i] = 1
        elif tr[i] == -1 and len(los) >= 2:
            (k1, l1), (k2, l2) = los[-2], los[-1]
            if l2 > l1 and i - k2 <= maxd:
                s = (l2 - l1) / (k2 - k1)
                if c[i] < l2 + s * (i - k2) and c[i - 1] >= l2 + s * (i - 1 - k2):
                    out[i] = -1
    return out


def flow_state(d: pd.DataFrame) -> dict:
    """今の流れ (大・中・細かい)"""
    i = len(d) - 1
    small = 1 if d.ema20.iat[i] > d.ema50.iat[i] else -1
    up, up4 = flows(d)
    return {"big": 1 if up[i] else -1, "mid": 1 if up4[i] else -1, "small": small,
            "aligned": int(trend(d)[i])}


def words(f: dict) -> str:
    w = lambda x: "上" if x == 1 else "下"
    al = {1: "上でそろう → 押し目で買う形を探す", -1: "下でそろう → 戻りで売る形を探す", 0: "大きな流れと中くらいの流れがそろわない → 入らない"}[f["aligned"]]
    return f"大きな流れ (3日) {w(f['big'])} / 中くらい (半日) {w(f['mid'])} / 15分足 {w(f['small'])} → {al}"
