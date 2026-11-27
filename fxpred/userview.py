"""利用者の相場観の記録と答え合わせ (利用者の案 2026-11-27: 定期的に私の相場観を報告する。参考にしてみて)。
記録: journal/user_views.csv (時刻・向き・上値・下値・シナリオ)。Claude への伝言、または GitHub の「相場観の登録」(my-view.yml) から。
答え合わせ: 24時間後・72時間後の値動きと、上値・下値に先に届いたのはどちらか。システム (流れ買いなど) の判断と並べて朝の報告に出す。
売買の判断には使わない (参考)。記録がたまったら、利用者の見立てと一致した時の作戦の成績を研究部が調べる。"""
import os

import pandas as pd

PATH = os.path.join(os.path.dirname(__file__), "..", "journal", "user_views.csv")
COLS = ["time_utc", "bias", "upper", "lower", "price", "scenario", "source"]


def add(bias: str, upper: float | None, lower: float | None, scenario: str, price: float | None = None, source="Claude", t=None):
    t = t or pd.Timestamp.now(tz="UTC")
    row = {"time_utc": f"{t:%Y-%m-%d %H:%M}", "bias": bias, "upper": upper, "lower": lower, "price": price,
           "scenario": scenario[:300], "source": source}
    old = pd.read_csv(PATH) if os.path.exists(PATH) else pd.DataFrame(columns=COLS)
    pd.concat([old, pd.DataFrame([row])]).to_csv(PATH, index=False)
    return row


def score(h: pd.DataFrame) -> pd.DataFrame:
    """24時間後・72時間後の動きと、上値・下値のどちらに先に届いたか"""
    if not os.path.exists(PATH):
        return pd.DataFrame()
    d = pd.read_csv(PATH)
    out = []
    for r in d.itertuples():
        t = pd.Timestamp(r.time_utc, tz="UTC")
        w = h[h.index > t]
        if w.empty:
            continue
        p0 = r.price if pd.notna(r.price) else float(h.Close.asof(t))
        sgn = 1 if "買" in str(r.bias) else -1 if "売" in str(r.bias) else 0
        m24 = (w.Close.iloc[min(23, len(w) - 1)] - p0) * 100 if len(w) >= 24 else None
        m72 = (w.Close.iloc[min(71, len(w) - 1)] - p0) * 100 if len(w) >= 72 else None
        first = "まだ"
        for hi, lo in zip(w.High, w.Low):
            up_hit = pd.notna(r.upper) and hi >= r.upper
            dn_hit = pd.notna(r.lower) and lo <= r.lower
            if up_hit or dn_hit:
                first = "上値" if up_hit and not dn_hit else "下値" if dn_hit and not up_hit else "同じ足"
                break
        out.append({"time_utc": r.time_utc, "bias": r.bias, "p0": round(p0, 3), "24h": m24, "72h": m72,
                    "hit24": (sgn * m24 > 0) if (sgn and m24 is not None) else None, "first": first})
    return pd.DataFrame(out)


def lines(h: pd.DataFrame) -> list[str]:
    s = score(h)
    if s.empty:
        return []
    last = s.iloc[-1]
    done = s.dropna(subset=["hit24"])
    out = [f"利用者の相場観 ({last.time_utc} UTC): {last.bias} / 先に届いたのは {last['first']}"
           + (f" / 24時間 {last['24h']:+.0f}pips" if pd.notna(last['24h']) else "")]
    if len(done):
        out.append(f"利用者の相場観の答え合わせ: {len(done)}回中 {int(done.hit24.sum())}回 当たり (24時間後の向き)")
    return out
