"""流れ × タイミングの通知 (利用者の手法 2026-11-27)。5分監視から呼ぶ。中身は flowtiming.py (研究と同じ関数)。
- 15分足が確定するたび (0・15・30・45分の直後) に、Yahoo の15分足 (60日) で流れを計算し直す
- 流れの変化 (大きな流れ・4時間足のどちらかが変わった) は、その場で知らせる (日本時間 10時〜翌1時)
- 流れがそろった中でトレンド転換ラインを抜けたら、入る合図を送る。15分以内に入る前提 (遅れるほど不利)"""
import numpy as np
import pandas as pd

KEY = "TL"


def _bars15() -> pd.DataFrame | None:
    try:
        import yfinance as yf
        x = yf.download("USDJPY=X", period="60d", interval="15m", progress=False, auto_adjust=False)
        x.columns = x.columns.get_level_values(0)
        x.index = pd.to_datetime(x.index, utc=True)
        x = x[["Open", "High", "Low", "Close"]].dropna()
        return x[x.index + pd.Timedelta(minutes=15) <= pd.Timestamp.now(tz="UTC")]     # 確定した足だけ
    except Exception as e:
        print("15分足を取れませんでした:", e)
        return None


def check(st: dict, now_epoch: int, price: float, dry=False) -> list[tuple[str, str, int]]:
    now = pd.Timestamp(now_epoch, unit="s", tz="UTC")
    if now.minute % 15 > 6:
        return []
    slot = f"{now.floor('15min'):%Y-%m-%d %H:%M}"
    if st.get("tl_slot") == slot:
        return []
    st["tl_slot"] = slot
    from . import explore as ex, flowtiming as ft, fmt, positions, scan, signal_audit
    raw = _bars15()
    if raw is None or len(raw) < 2500:
        return []
    d = ex.bars(tf=None, src=raw.copy())
    d = d[d.atr > 0]
    out = []
    jh = (now + pd.Timedelta(hours=9)).hour
    active = jh >= 10 or jh < 1
    # 流れの変化
    f = ft.flow_state(d)
    prev = st.get("tl_flow")
    st["tl_flow"] = f
    if prev and (prev["big"] != f["big"] or prev["mid"] != f["mid"]) and active:
        ch = []
        if prev["big"] != f["big"]:
            ch.append(f"大きな流れ (3日) が {'上→下' if f['big'] == -1 else '下→上'}")
        if prev["mid"] != f["mid"]:
            ch.append(f"中くらいの流れ (半日) が {'上→下' if f['mid'] == -1 else '下→上'}")
        out.append((f"🔄 流れの変化 USD/JPY {price:.3f}", " / ".join(ch) + f"\n今の流れ: {ft.words(f)}", 9807270))
    # 入る合図 (最後に確定した15分足)
    sig = ft.trendline_signal(d)
    s = int(sig[-1])
    print(f"(流れ×タイミング) {ft.words(f)} / 合図 {s}")
    if not s or not active or KEY not in scan.notify_keys():
        return out
    a = float(d.atr.iat[-1]); c = float(d.Close.iat[-1])
    stop = c - s * 1.0 * a
    if abs(price - c) > 0.5 * a:      # 足の確定から大きく動いた後なら入らない (遅れに弱い)
        scan.log_signal(now.floor("min"), KEY, {"side": s, "entry": price, "stop": stop, "target": None}, 0, "見送り(動いた後)", sent=False, conditions="遅れ")
        return out
    if any(p.get("key") == KEY and p["side"] == s for p in positions.load()):
        return out
    blk, notes = signal_audit.gate(s, key=KEY)
    cb, ct = signal_audit.cost_gate(price, stop)
    if blk or cb:
        print("(流れ×タイミング) 止めた:", notes if blk else ct)
        if not dry:
            scan.log_signal(now.floor("min"), KEY, {"side": s, "entry": price, "stop": stop, "target": None}, 0, "見送り(関門)", sent=False, conditions="関門")
        return out
    from .org import infodesk
    reasons = [f"流れ: {ft.words(f)}", "押し目 (戻り) のトレンド転換ラインを15分足で抜けた", infodesk.fundamentals_note(s, KEY)]
    cautions = ["新しい作戦 (試し)。量は小さめに", f"15分以内に入る。{c + s * 0.5 * a:.3f} より不利なら見送り (遅れるほど成績が下がる)", ct] + notes
    title, body = fmt.signal(s, price, stop, None, 0, reasons, cautions, f"{(now + pd.Timedelta(hours=9)):%m/%d %H:%M} から15分以内", None, key=KEY)
    if not dry:
        scan.log_signal(now.floor("min"), KEY, {"side": s, "entry": price, "stop": stop, "target": None}, 28, "買い" if s == 1 else "売り",
                        sent=True, conditions="流れ×タイミング")
        positions.add(s, price, stop, price + s * 50.0, KEY, int(now.timestamp()), atr=a)
    out.append((title, body, 5763719))
    return out
