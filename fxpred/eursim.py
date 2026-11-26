"""ユーロドル・ユーロ円で、システムが利益を出せるかの検証 (裏で回す。GitHub Actions の eur-sim、公開リポジトリなので無料。AI は使わない)。

試すこと (ペアごと、1時間足と 15分足):
 1. ドル円と同じ総当たり (合図 × 絞り込み × 決済 × 時間帯)。ドル円で使っている作戦 (急変に付いていく・もみ合いのブレイク・MACD など) も含む
 2. ドル・ユーロ・円の3つの強さの作戦 (強さの差・3つの並び・急変・誰が動かした急変か)
 3. 偽物の相場 (20日ごとに順番を入れ替え、3ペア同じ入れ替え) で同じことをして、偶然でも同じくらい通るか比べる
作る期間 / 確かめる期間: 1時間足は 2004-2015 / 2016-。1分足 (15分足) は手元の年の前半 / 後半 (2004年からそろえば 2004-2015 / 2016-)。
往復コスト (SBI より少し厳しめ): ユーロドル 0.5pips / ユーロ円 1.0pips。ユーロドルは価格を100倍して pips をドル円と同じ 0.01 にそろえる。
結果: reports/eur_sim.md (通知候補 = 厳しめの基準を満たし、成績が偽物の相場の最大を超えたもの) / journal/eur_sim.csv (回ごとのまとめ)"""
import glob
import lzma
import os
import sys
import time

import numpy as np
import pandas as pd

from . import explore as ex
from . import m1data

ROOT = os.path.join(os.path.dirname(__file__), "..")
PAIRS_DIR = os.path.join(ROOT, "data", "pairs")
COST = {"EURUSD": 0.5, "EURJPY": 1.0}
SCALE = {"EURUSD": 100.0, "EURJPY": 1.0, "USDJPY": 1.0}
NAME = {"EURUSD": "ユーロドル", "EURJPY": "ユーロ円"}


def hourly(pair) -> pd.DataFrame:
    """data-pairs ブランチの1時間足 (2004年〜)。USDJPY は data-duka"""
    parts = []
    for f in sorted(glob.glob(os.path.join(PAIRS_DIR, "data", "duka", f"{pair}_*_BID_hour.bi5"))):
        b = open(f, "rb").read()
        if not b:
            continue
        y, m = map(int, os.path.basename(f).split("_")[1:3])
        r = np.frombuffer(lzma.decompress(b), dtype=">i4,>i4,>i4,>i4,>i4,>f4")
        sc = 1e5 if pair == "EURUSD" else 1e3
        parts.append(pd.DataFrame({"Open": r["f1"] / sc, "Close": r["f2"] / sc, "Low": r["f3"] / sc, "High": r["f4"] / sc},
                                  index=pd.Timestamp(year=y, month=m, day=1, tz="UTC") + pd.to_timedelta(r["f0"], unit="s")))
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts).sort_index()
    return d[~d.index.duplicated() & (d.High > d.Low)]


def fake(frames: dict, seed: int) -> dict:
    """偽物の相場: 20日分 (1時間足480本ぶん) の塊ごとに順番を入れ替える。全ペア同じ入れ替え"""
    idx = None
    for d in frames.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    n = len(idx)
    per_day = max(int(round(pd.Timedelta("1D") / pd.Series(idx[1:] - idx[:-1]).median())), 1)
    B = 20 * per_day
    rng = np.random.default_rng(seed)
    starts = np.arange(0, n, B)
    pos = np.concatenate([np.arange(s, min(s + B, n)) for s in starts[rng.permutation(len(starts))]])
    out = {}
    for k, d in frames.items():
        d = d.loc[idx]
        lc = np.log(d.Close.values)
        r = np.diff(lc, prepend=lc[0])
        rel = np.log(d[["Open", "High", "Low"]].values) - lc[:, None]
        nc = np.cumsum(r[pos]) + lc[0]
        out[k] = pd.DataFrame(np.exp(np.c_[rel[pos] + nc[:, None], nc]), index=idx, columns=["Open", "High", "Low", "Close"])
    return out


def strength_signals(d: pd.DataFrame, frames: dict, pair: str) -> dict:
    """3つの強さの作戦の合図 (ex.search の sig の形: 名前 → (種類, 配列))"""
    al = {k: v.reindex(d.index).Close for k, v in frames.items()}
    if any(v.isna().mean() > 0.2 for v in al.values()):
        return {}
    lEU, lEJ, lUJ = (np.log(al[k].ffill()) for k in ("EURUSD", "EURJPY", "USDJPY"))
    per_day = d.attrs.get("per_day", 24)
    a_, b_ = ("e", "u") if pair == "EURUSD" else ("e", "j")
    other = "j" if b_ == "u" else "u"
    S = {}
    for hours in (4, 24, 72):
        n = max(int(hours * per_day / 24), 1)
        rEU, rEJ, rUJ = lEU.diff(n), lEJ.diff(n), lUJ.diff(n)
        st = {"e": (rEU + rEJ) / 3, "u": (rUJ - rEU) / 3, "j": -(rUJ + rEJ) / 3}
        st = {k: (v / v.rolling(50 * per_day, min_periods=10 * per_day).std()).values for k, v in st.items()}
        diff = st[a_] - st[b_]
        for k in (2.0, 3.0):
            sig = np.where(diff >= k, 1, np.where(diff <= -k, -1, 0))
            S[f"強さの差{hours}h {k}倍 順張り"] = ("強さ", sig)
            S[f"強さの差{hours}h {k}倍 逆張り"] = ("強さ", -sig)
        A_, B_, X_ = st[a_], st[b_], st[other]
        for k in (1.5, 2.0):
            up = (A_ >= k) & (B_ <= -k) & (A_ > X_) & (X_ > B_)
            dn = (A_ <= -k) & (B_ >= k) & (A_ < X_) & (X_ < B_)
            S[f"3つの並び{hours}h {k}倍"] = ("強さ", np.where(up, 1, np.where(dn, -1, 0)))
    n3 = max(int(3 * per_day / 24), 1)
    lp = np.log(d.Close)
    r3 = lp.diff(n3)
    z3 = (r3 / r3.rolling(50 * per_day, min_periods=10 * per_day).std()).values
    rEU, rEJ, rUJ = lEU.diff(n3), lEJ.diff(n3), lUJ.diff(n3)
    m3 = np.argmax(np.abs(np.vstack([((rEU + rEJ) / 3).values, ((rUJ - rEU) / 3).values, (-(rUJ + rEJ) / 3).values])), axis=0)
    code = {"e": 0, "u": 1, "j": 2}
    nm = {"e": "ユーロ", "u": "ドル", "j": "円"}
    for k in (3.0, 4.0):
        sig = np.where(z3 >= k, 1, np.where(z3 <= -k, -1, 0))
        for who in (a_, b_, other):
            m = m3 == code[who]
            S[f"急変{k}倍 {nm[who]}が動かした 付いていく"] = ("強さ", np.where(m, sig, 0))
            S[f"急変{k}倍 {nm[who]}が動かした 戻り"] = ("強さ", np.where(m, -sig, 0))
    return {k: (kind, np.nan_to_num(v).astype(np.int64)) for k, (kind, v) in S.items()}


def one(pair, tf, frames, build_end):
    """1つのペア・足で総当たり (普通の作戦 + 3つの強さ)"""
    src = frames[pair] * SCALE[pair]
    d = ex.bars(tf=None, src=src.copy())        # frames はもうその足の長さ
    d = d[d.atr > 0]
    ex.BUILD_END = build_end
    sig = ex.signals(d)
    sig.update(strength_signals(d, frames, pair))
    res, _ = ex.search(d, tf, cost_pips=COST[pair], windows=("10時〜翌1時", "いつでも"), sig=sig)
    res["ペア"] = NAME[pair]
    return res


def strict(r: pd.DataFrame) -> pd.DataFrame:
    return r[(r.作る_R > 0.05) & (r.確かめ_R > 0.05) & (r.年間 >= 5) & (r.確かめ_回数 >= 30)
             & (r.プラス年代 >= 0.6 * r.年代数)]


def main(fakes=2):
    t0 = time.time()
    lines = [f"# ユーロドル・ユーロ円の検証 ({pd.Timestamp.now(tz='Asia/Tokyo'):%Y-%m-%d %H:%M} 日本時間)", "",
             "厳しめの基準: 作る・確かめるの両方で +0.05R 超、年5回以上、時代の6割以上でプラス。"
             "**通知候補** = 基準を満たし、確かめる期間の成績が偽物の相場で一番良かったものより上。", ""]
    summary = []
    sets = []
    h = {p: hourly(p) for p in ("EURUSD", "EURJPY", "USDJPY")}
    if all(len(v) for v in h.values()):
        sets.append(("1h", h, 2015))
    m1 = {p: m1data.load(pair=p) for p in ("EURUSD", "EURJPY", "USDJPY")}
    if all(len(v) for v in m1.values()):
        start = max(v.index[0] for v in m1.values()).year
        end = min(v.index[-1] for v in m1.values()).year
        be = 2015 if start <= 2006 else (start + end) // 2
        f15 = {p: m1data.resample(v[(v.index.year >= start)], "15min") for p, v in m1.items()}
        sets.append(("15min", f15, be))
        lines.append(f"1分足: {start}〜{end}年 (15分足にまとめて使う。作る ~{be} / 確かめ {be + 1}~)")
    else:
        lines.append("1分足: ユーロのペアがまだそろっていない (1時間足だけで検証)")
    lines.append("")
    for tf, frames, be in sets:
        for pair in ("EURUSD", "EURJPY"):
            real = one(pair, tf, frames, be)
            fk = []
            for s in range(1, fakes + 1):
                ff = fake(frames, s)
                r = one(pair, tf, ff, be)
                fk.append((len(strict(r)), r.確かめ_R[strict(r).index].max() if len(strict(r)) else np.nan, (r.作る_R > 0).mul(r.確かめ_R > 0).mean()))
            st = strict(real)
            fmax = np.nanmax([x[1] for x in fk]) if any(not np.isnan(x[1]) for x in fk) else 0.0
            cand = st[st.確かめ_R > fmax].sort_values("確かめ_R", ascending=False)
            lines += [f"## {NAME[pair]} {tf} (作る ~{be} / 確かめ {be + 1}~)",
                      f"- 試した数 {len(real)}、両方プラス {((real.作る_R > 0) & (real.確かめ_R > 0)).mean():.0%} "
                      f"(偽物 {', '.join(f'{x[2]:.0%}' for x in fk)})",
                      f"- 厳しめの基準を満たした数 {len(st)} (偽物 {', '.join(str(x[0]) for x in fk)})、偽物の確かめの最大 {fmax:+.2f}R",
                      f"- 通知候補 {len(cand)}個"]
            top = (cand if len(cand) else st.sort_values("確かめ_R", ascending=False)).head(8)
            for r in top.itertuples():
                lines.append(f"  - {'★' if r.Index in cand.index else '・'} {r.合図} / {r.絞り込み} / {r.決済} / {r.時間帯}: "
                             f"年{r.年間:.0f}回 勝率{r.勝率:.0%} 作る {r.作る_R:+.2f}R 確かめ {r.確かめ_R:+.2f}R (プラスの時代 {r.プラス年代}/{r.年代数})")
            # ドル円で使っている作戦がユーロでどうか
            usual = real[real.合図.str.contains("急な動きに付いていく|もみ合い後のブレイク|MACD")]
            if len(usual):
                g = usual.groupby("合図")[["作る_R", "確かめ_R"]].mean()
                lines.append("- ドル円で使っている形の平均: " + " / ".join(f"{k} 作る{v.作る_R:+.2f} 確かめ{v.確かめ_R:+.2f}" for k, v in g.iterrows()))
            lines.append("")
            summary.append(dict(time=f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M}", pair=NAME[pair], tf=tf, tested=len(real), strict=len(st),
                                fake_strict="/".join(str(x[0]) for x in fk), fake_max=round(float(fmax), 3), candidates=len(cand),
                                best=(f"{cand.iloc[0].合図}|{cand.iloc[0].絞り込み}|{cand.iloc[0].決済}|{cand.iloc[0].時間帯}" if len(cand) else ""),
                                best_verify=round(float(cand.iloc[0].確かめ_R), 3) if len(cand) else np.nan))
            print(f"{NAME[pair]} {tf}: 基準 {len(st)} 偽物 {[x[0] for x in fk]} 候補 {len(cand)} ({time.time() - t0:.0f}s)", flush=True)
    lines += ["## 判断のしかた", "",
              "- 通知候補が出ても、すぐには通知しない。影の運用で記録し、成績を確かめてから利用者と相談する",
              "- ユーロの検証と見張りは AI の無料枠を使わず、ドル円の監視とは別に動く。ドル円の精度に影響しないよう、"
              "ユーロの処理でエラーや遅れが増えたら `config/pairs.json` の eur を false にして止める"]
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    open(os.path.join(ROOT, "reports", "eur_sim.md"), "w").write("\n".join(lines) + "\n")
    path = os.path.join(ROOT, "journal", "eur_sim.csv")
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    pd.concat([old, pd.DataFrame(summary)]).to_csv(path, index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main(fakes=int(sys.argv[1]) if len(sys.argv) > 1 else 2)
