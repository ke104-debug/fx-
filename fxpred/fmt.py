"""通知の見た目 (視認性重視)。売買の判断・エントリー・損切り・利確・信頼度を必ず先頭に置く"""


def pips(a, b, side=1):
    """ポジションから見た損益 (pips)。損切りは常にマイナス、利確は常にプラスになる"""
    return round(side * (b - a) * 100)


import re


def clean(x: str) -> str:
    """条件の説明を短い言葉に"""
    x = x.split("、ただし")[0]
    rules = [(r"^時間足 \((.+)が(上昇|下降)\)$", r"\1が\2"),
             (r"^時間足 \((上昇|下降)の時間足なし.*\)$", "時間足の方向がそろっていない"),
             (r"^時間足 \((.+)\)$", r"時間足: \1"),
             (r"^価格帯 \(重要な価格帯が近くにない\)$", "下値・上値から離れた位置"),
             (r"^価格帯 \((.+)\)$", r"\1"),
             (r"^市場心理 \(VIX 低下.*\)$", "VIX低下でリスク回避が和らぐ"),
             (r"^市場心理 \(VIX 上昇.*\)$", "VIX上昇でリスク回避が強まる"),
             (r"^AIモデル \((.+?)\)?$", r"AIモデル: \1")]
    for pat, rep in rules:
        if re.match(pat, x):
            return re.sub(pat, rep, x)
    return x


def signal(side: int, entry: float, stop: float, target: float | None, conf: int, reasons: list[str],
           cautions: list[str], valid: str, no_trade: list[str] | None = None) -> tuple[str, str]:
    if target is None:                     # 利確を決めない作戦でも、目安は必ず出す (損切り幅の2倍)
        target = entry + side * 2 * abs(entry - stop)
    word = "買い" if side == 1 else "売り"
    mark = "🟢" if side == 1 else "🔴"
    if no_trade:
        title = f"⚪ 見送り推奨 USD/JPY ({word}の条件は成立)"
    else:
        title = f"{mark} {word} USD/JPY 信頼度 {conf}%"
    lo, hi = sorted((entry - 0.05, entry + 0.05))
    rr = abs(target - entry) / max(abs(entry - stop), 1e-9)
    lines = [
        f"**判断　　: {'見送り' if no_trade else word}**",
        f"**エントリー: {lo:.2f} 〜 {hi:.2f}**",
        f"**損切り　: {stop:.2f}**（{pips(entry, stop, side):+d} pips）",
        f"**利確　　: {target:.2f}**（{pips(entry, target, side):+d} pips）",
        f"**信頼度　: {conf}%**　損益比 1 : {rr:.1f}",
        f"**遅れた時: {entry + side * max(0.05, 0.3 * abs(entry - stop)):.2f} より不利なら入らない**（通知から60分を過ぎたら見送り）",
        "━━━━━━━━━━",
    ]
    if no_trade:
        lines += ["見送りの理由"] + [f"・{clean(x)}" for x in no_trade[:3]] + [""]
    lines += ["理由"] + [f"・{clean(x)}" for x in reasons[:3]]
    if cautions:
        lines += ["", "注意"] + [f"・{clean(x)}" for x in cautions[:3]]
    lines += ["", f"有効: {valid}", "※AIの分析であり利益を保証しません"]
    return title, "\n".join(lines)
