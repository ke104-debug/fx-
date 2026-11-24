"""相場分析部の専門AI (1時間足で毎時意見を出す)。⑩テクニカル ⑪ローソク足・パターン ⑫サポレジ ⑬マルチタイムフレーム ⑭ボラティリティ ⑮市場構造。
どれも「その足が確定した時点」で分かる値だけを使う (未来の足を見ない)。意見は -1 (売り寄り) 〜 +1 (買い寄り)。
向きは「よく言われる使い方」で固定し、検証部 (score.py の hourly) が4時間先・24時間先で採点して重みを決める。"""
import numpy as np
import pandas as pd


def _rsi(c, n=14):
    d = c.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / n).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n).mean()
    return 100 - 100 / (1 + up / dn)


def _atr(h, n=14):
    c = h.Close
    tr = pd.concat([h.High - h.Low, (h.High - c.shift()).abs(), (h.Low - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _pivots(h, w=3):
    """確定したスイング高値・安値 (w 本後に確定する → w 本遅らせて使う)"""
    hi, lo = h.High, h.Low
    ph = (hi == hi.rolling(2 * w + 1, center=True).max())
    pl = (lo == lo.rolling(2 * w + 1, center=True).min())
    # 中心の足が高値と分かるのは w 本後 → その時点の値として w 本ずらす
    return ph.shift(w, fill_value=False).astype(bool), pl.shift(w, fill_value=False).astype(bool), None, None


def features(h: pd.DataFrame) -> pd.DataFrame:
    c, o, hi, lo = h.Close, h.Open, h.High, h.Low
    A = _atr(h)
    b = pd.DataFrame(index=h.index)
    # ---- ⑩ テクニカル分析AI
    rsi = _rsi(c)
    b["テクニカル|RSIの逆張り(30/70)"] = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
    b["テクニカル|RSIの向き(50より上なら買い)"] = ((rsi - 50) / 20).clip(-1, 1)
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    mh = macd - macd.ewm(span=9).mean()
    b["テクニカル|MACDの向き"] = np.sign(mh)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    z = (c - mid) / sd
    b["テクニカル|ボリンジャーの逆張り(2σ)"] = np.where(z < -2, 1, np.where(z > 2, -1, 0))
    b["テクニカル|移動平均の並び(20>50>200)"] = np.sign((c.ewm(span=20).mean() > c.ewm(span=50).mean()).astype(int)
                                               + (c.ewm(span=50).mean() > c.ewm(span=200).mean()).astype(int) - 1)
    k = 100 * (c - lo.rolling(14).min()) / (hi.rolling(14).max() - lo.rolling(14).min())
    st = k.rolling(3).mean()
    b["テクニカル|ストキャスの逆張り(20/80)"] = np.where(st < 20, 1, np.where(st > 80, -1, 0))
    tech = b[[x for x in b if x.startswith("テクニカル")]]
    b["テクニカル|総合評価(多数決)"] = np.sign(tech.sum(axis=1))
    # ---- ⑪ ローソク足・チャートパターン分析AI
    body, rng = (c - o), (hi - lo).replace(0, np.nan)
    po, pc = o.shift(), c.shift()
    bull_eng = (c > o) & (pc < po) & (c >= po) & (o <= pc)
    bear_eng = (c < o) & (pc > po) & (c <= po) & (o >= pc)
    b["ローソク足|包み足"] = np.where(bull_eng, 1, np.where(bear_eng, -1, 0))
    lw, uw = (np.minimum(c, o) - lo), (hi - np.maximum(c, o))
    pin_up = (lw >= 2 * body.abs()) & (lw >= 0.6 * rng) & (lo <= lo.rolling(24).min())
    pin_dn = (uw >= 2 * body.abs()) & (uw >= 0.6 * rng) & (hi >= hi.rolling(24).max())
    b["ローソク足|ピンバー(安値・高値で)"] = np.where(pin_up, 1, np.where(pin_dn, -1, 0))
    ph, pl, _, _ = _pivots(h)
    tol = 0.3 * A

    def piv(vals: pd.Series, mask: pd.Series, k: int) -> pd.Series:
        """確定したスイングの値の k 個前 (0 = 最新) を、各足の時点の値として並べる"""
        ev = vals[mask].shift(k)
        return ev.reindex(h.index).ffill()
    lows, lows_prev, lows_prev2 = (piv(lo.shift(3), pl, k) for k in (0, 1, 2))
    highs, highs_prev, highs_prev2 = (piv(hi.shift(3), ph, k) for k in (0, 1, 2))
    dbl_bot = ((lows - lows_prev).abs() < tol) & (c > highs) & (c.shift() <= highs.shift())
    dbl_top = ((highs - highs_prev).abs() < tol) & (c < lows) & (c.shift() >= lows.shift())
    b["チャートパターン|ダブルボトム・トップ"] = np.where(dbl_bot, 1, np.where(dbl_top, -1, 0))
    # 三尊/逆三尊 (簡易): 高値3つで真ん中が一番高く、両肩がほぼ同じ → 直近の安値を割ったら売り
    h1, h2, h3 = highs_prev2, highs_prev, highs
    hs = (h2 > h1 + tol) & (h2 > h3 + tol) & ((h1 - h3).abs() < 2 * tol) & (c < lows) & (c.shift() >= lows.shift())
    l1, l2, l3 = lows_prev2, lows_prev, lows
    ihs = (l2 < l1 - tol) & (l2 < l3 - tol) & ((l1 - l3).abs() < 2 * tol) & (c > highs) & (c.shift() <= highs.shift())
    b["チャートパターン|三尊・逆三尊"] = np.where(ihs, 1, np.where(hs, -1, 0))
    # 三角持ち合い・フラッグ: 24本の値幅が縮んだ後 (前の24本の半分以下) の抜け
    w24, wprev = hi.rolling(24).max() - lo.rolling(24).min(), (hi.rolling(24).max() - lo.rolling(24).min()).shift(24)
    sq = w24 < 0.5 * wprev
    b["チャートパターン|持ち合い(三角・フラッグ)の抜け"] = np.where(sq.shift() & (c > hi.rolling(24).max().shift()), 1,
                                                np.where(sq.shift() & (c < lo.rolling(24).min().shift()), -1, 0))
    # ---- ⑫ サポート・レジスタンス分析AI
    sup, res = lo.rolling(120).min().shift(), hi.rolling(120).max().shift()
    b["サポレジ|支持線で反発・抵抗線で反落"] = np.where((lo <= sup + tol) & (c > sup + tol), 1,
                                             np.where((hi >= res - tol) & (c < res - tol), -1, 0))
    b["サポレジ|ブレイクアウト(120本)"] = np.where(c > res, 1, np.where(c < sup, -1, 0))
    rn = (c * 2).round() / 2      # 0.5円刻みのキリ番
    b["サポレジ|キリ番の手前で反発"] = np.where((lo <= rn) & (c > rn) & (rn < c.shift()), 1,
                                         np.where((hi >= rn) & (c < rn) & (rn > c.shift()), -1, 0))
    up_tr = c.ewm(span=480).mean().diff(24) > 0
    pull = (c - c.rolling(48).max()) / A
    b["サポレジ|押し目買い・戻り売り(大きな流れの方向)"] = np.where(up_tr & (pull < -2), 1,
                                                    np.where(~up_tr & ((c - c.rolling(48).min()) / A > 2), -1, 0))
    # ---- ⑬ マルチタイムフレーム分析AI (1時間足から 4時間・日足相当を作る。1分〜15分足は短い足の研究で別に)
    s1 = np.sign(c.ewm(span=20).mean().diff())
    s4 = np.sign(c.ewm(span=80).mean().diff(4))
    sd_ = np.sign(c.ewm(span=480).mean().diff(24))
    b["マルチタイム|1時間・4時間・日足がそろう"] = np.where((s1 == s4) & (s4 == sd_), s1, 0)
    b["マルチタイム|上位足の向きで下位足の押し目"] = np.where((sd_ == 1) & (s4 == 1) & (s1 == -1), 1,
                                                np.where((sd_ == -1) & (s4 == -1) & (s1 == 1), -1, 0))
    # ---- ⑭ ボラティリティ分析AI (向きは出さず、他の意見の強さを変える役。ここでは「値動きが大きい時の勢い」に付く)
    ar = A / A.rolling(480, min_periods=100).mean()
    mom = np.sign(c.diff(4))
    b["ボラティリティ|値動き拡大時の勢いに付く"] = np.where(ar > 1.5, mom, 0)
    b["ボラティリティ|値動き縮小後の最初の大きな足"] = np.where((ar.shift() < 0.7) & (rng > 2 * A.shift()), np.sign(body), 0)
    # ---- ⑮ 市場構造分析AI
    # BOS (構造の継続): 上昇の流れで直近のスイング高値を上抜け / CHoCH (転換): 下降の流れで直近のスイング高値を上抜け
    hh = (highs > highs_prev) & (lows > lows_prev)
    ll = (highs < highs_prev) & (lows < lows_prev)
    brk_up = (c > highs) & (c.shift() <= highs.shift())
    brk_dn = (c < lows) & (c.shift() >= lows.shift())
    b["市場構造|BOS(流れの継続ブレイク)"] = np.where(hh & brk_up, 1, np.where(ll & brk_dn, -1, 0))
    b["市場構造|CHoCH(流れの転換ブレイク)"] = np.where(ll & brk_up, 1, np.where(hh & brk_dn, -1, 0))
    # ストップ狩り: 直近の安値 (高値) を一度割ってすぐ戻した足 → 逆に動きやすい
    b["市場構造|ストップ狩り後の戻り"] = np.where((lo < lows) & (c > lows), 1, np.where((hi > highs) & (c < highs), -1, 0))
    return b.astype(float)
