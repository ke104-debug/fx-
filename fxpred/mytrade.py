"""利用者が自分で入った取引を登録し、損切り・利確・損切りの引き上げを5分監視で知らせる。
GitHub の「自分の取引を登録」(my-trade.yml) から呼ばれる。リポジトリは公開なので、記録に残るのは向きと価格だけ (数量・金額は扱わない)。

損切りを空欄にした時の決め方 (決済・ポジション管理部隊):
  直近12時間の安値 (売りなら高値) の少し外 (0.2ATR)。ただし 0.8〜2.5ATR の範囲に収める
利確の目安: 損切り幅の3倍 (損小利大)。ただし固定の利確では決済せず、1R 進んだら損切りを建値へ、その後は最も伸びた値から2ATR後ろに引き上げ続ける
  (23年の検証で、伸ばす決済が固定の利確より良かった作戦と同じ管理)。期限は7日。
python -m fxpred.mytrade register 買い 159.02 [損切り]
python -m fxpred.mytrade close [決済価格]
python -m fxpred.mytrade list"""
import json
import os
import sys
import time

import pandas as pd

from . import notify, positions, regime as rg, scan


def _bars():
    return rg.prepare(scan.load_recent())


def plan(side: int, entry: float, stop: float | None, h: pd.DataFrame) -> dict:
    a = float(h.atr.iat[-1])
    if stop is None:
        sw = (h.Low.iloc[-12:].min() - 0.2 * a) if side == 1 else (h.High.iloc[-12:].max() + 0.2 * a)
        dist = min(max(abs(entry - sw), 0.8 * a), 2.5 * a)
        stop, how = entry - side * dist, "直近12時間の安値の少し外" if side == 1 else "直近12時間の高値の少し外"
    else:
        how = "入力した損切り"
    risk = abs(entry - stop)
    return {"stop": round(stop, 3), "target": round(entry + side * 3 * risk, 3), "atr": a, "risk_pips": risk * 100, "how": how}


def register(side_word: str, price: str, stop: str = "") -> str:
    side = 1 if side_word.strip() in ("買い", "買", "buy", "BUY", "1") else -1
    entry = float(price)
    h = _bars()
    st = float(stop) if stop and stop.strip() else None
    if st is not None and side * (entry - st) <= 0:
        raise SystemExit("損切りが入った値段の反対側にありません (買いなら下、売りなら上)")
    p = plan(side, entry, st, h)
    now = int(time.time())
    positions.add(side, entry, p["stop"], p["target"], "U", now, atr=p["atr"])
    word = "買い" if side == 1 else "売り"
    lines = [f"**{word} {entry:.3f}** を登録しました",
             f"損切り **{p['stop']:.3f}** ({p['risk_pips']:.0f}pips・{p['how']})",
             f"利確の目安 **{p['target']:.3f}** (損切り幅の3倍)",
             "・1R (損切り幅と同じだけ) 進んだら損切りを建値へ、その後は最も伸びた値から2ATR後ろに引き上げ続けます (その都度通知)",
             "・損切り・引き上げた損切りに届いた時、7日の期限、週末前に通知します",
             "・損切りは注文 (逆指値) で入れておくのがおすすめです。引き上げの通知が来たら注文を直してください"]
    try:   # 参考: 組織の今の見立て
        from .org import meeting
        res = meeting.final(h)
        ov = res["overall"]
        rel = "同じ向き" if ov * side >= 0.25 else "逆向き" if ov * side <= -0.25 else "中立"
        lines += ["━━━━━━━━━━", f"参考: 方向の会議は{rel} ({res['verdict']} {ov:+.2f}) / リスク {res['risk']['level']}"
                  + ("".join(f"\n・{w}: {y}" for w, y in res["risk"]["items"][:3]))]
    except Exception as e:
        print("組織の見立てを付けられませんでした:", e)
    body = "\n".join(lines)
    _out(f"📝 登録: {word} USD/JPY {entry:.3f}", body, 3447003)
    return body


def close(price: str = "") -> str:
    ps = positions.load()
    mine = [p for p in ps if p.get("key") == "U"]
    if not mine:
        _out("📝 決済の登録", "登録中の自分の取引はありません", 9807270)
        return "なし"
    p = mine[-1]
    px = float(price) if price and price.strip() else float(_bars().Close.iat[-1])
    positions._record(p, px, "自分で決済", int(time.time()))
    positions.save([q for q in ps if q["id"] != p["id"]])
    pips = p["side"] * (px - p["entry"]) * 100
    body = f"{'買い' if p['side'] == 1 else '売り'} {p['entry']:.3f} → {px:.3f} ({pips:+.0f}pips)。見張りを終えました"
    _out("📝 決済を登録 USD/JPY", body, 5763719 if pips >= 0 else 15548997)
    return body


def show() -> str:
    mine = [p for p in positions.load() if p.get("key") == "U"]
    body = "\n".join(f"・{'買い' if p['side'] == 1 else '売り'} {p['entry']:.3f} 損切り {p['stop']:.3f} 利確目安 {p['target']:.3f}" for p in mine) or "登録中の取引はありません"
    _out("📝 登録中の自分の取引", body, 9807270)
    return body


OUT = "state/mytrade_msg.json"


def _out(title, body, color):
    """送る内容をいったん保存する (保存 (git push) が成功してから notify_last で送る)"""
    os.makedirs("state", exist_ok=True)
    json.dump({"title": title, "body": body, "color": color}, open(OUT, "w"), ensure_ascii=False)
    print(title + "\n" + body)


def notify_last(cmd="list", *_):
    if cmd == "list":
        show()
    if os.path.exists(OUT):
        m = json.load(open(OUT))
        notify.send(m["title"], m["body"], m["color"])


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["list"]
    print({"register": register, "close": close, "list": show, "notify_last": notify_last}[cmd](*args))
