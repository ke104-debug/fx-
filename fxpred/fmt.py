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


# 作戦ごとの利確・決済のタイミング (23年の検証で使った決済そのもの。2026-11-27 利用者の要望: チャンス通知に利確のタイミングを必ず付ける)
# 転換ラインでの早めの利確・途中で損切りを建値に動かすことは、どの作戦でも成績が下がったので勧めない
EXIT_PLAN = {
    "SF": ["利確: {target:.2f} に届いたら (損切り幅の2倍)。注文 (IFD-OCO) で置いておく",
           "届かなければ 24時間後に、損益にかかわらず決済 (通知します)",
           "途中で損切りは動かさない (建値に動かすと23年の成績が下がる)"],
    "S": ["利確の値は置かずに伸ばす",
          "1R (損切り幅と同じだけ) 進んだら損切りを建値へ、その後は最も伸びた値から2ATR後ろへ引き上げ (その都度通知)",
          "引き上げた損切りに届いたら決済 (通知します)。最長9日"],
    "TF": ["利確の値は置かずに伸ばす (数日〜数週間)",
           "最も伸びた値から 5×日足ATR 後ろへ損切りを引き上げ (10pips 以上上げたら通知)",
           "毎朝10時に流れが下向きに変わったら決済 (通知します)"],
    "TL": ["利確の値は置かずに伸ばす (最長24時間)",
           "1R (損切り幅と同じだけ) 進んだら損切りを建値へ、その後は最も伸びた値から2ATR (15分足) 後ろへ引き上げ (その都度通知)",
           "引き上げた損切りか24時間で決済 (通知します)"],
}


def _stats(key):
    try:
        import json, os
        return json.load(open(os.path.join(os.path.dirname(__file__), "..", "config", "strategy_stats.json")))["stats"].get(key)
    except Exception:
        return None


def signal(side: int, entry: float, stop: float, target: float | None, conf: int, reasons: list[str],
           cautions: list[str], valid: str, no_trade: list[str] | None = None, key: str | None = None) -> tuple[str, str]:
    # 信頼度 = その作戦の23年の勝率 (2026-11-27 K003: 以前の点数は2.8年のデータで決めた配点で、テクニカルを予想に使っていた)
    ss = _stats(key) if key else None
    conf_note = ""
    if ss:
        conf = int(round(ss["win"] * 100))
        conf_note = f"（23年の勝率。{ss['n']}回・1回平均 {ss['R']:+.2f}R。損切りより利益を大きく取るので5割未満でもプラス）"
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
        (f"**利確　　: 置かない (伸ばす)**" if key in ("S", "TF", "TL") else f"**利確　　: {target:.2f}**（{pips(entry, target, side):+d} pips）"),
        f"**信頼度　: {conf}%**　損益比 1 : {rr:.1f}" + conf_note,
        f"**遅れた時: {entry + side * max(0.05, 0.3 * abs(entry - stop)):.2f} より不利なら入らない**（通知から60分を過ぎたら見送り）",
        "━━━━━━━━━━",
    ]
    if no_trade:
        lines += ["見送りの理由"] + [f"・{clean(x)}" for x in no_trade[:3]] + [""]
    lines += ["理由"] + [f"・{clean(x)}" for x in reasons[:3]]
    if cautions:
        lines += ["", "注意"] + [f"・{clean(x)}" for x in cautions[:3]]
    if key in EXIT_PLAN:   # 利確・決済のタイミング (検証した決済そのもの)
        lines += ["", "利確・決済のタイミング"] + [f"・{x.format(target=target, stop=stop)}" for x in EXIT_PLAN[key]]
    lines += ["", f"有効: {valid}", "※AIの分析であり利益を保証しません"]
    return title, "\n".join(lines)
