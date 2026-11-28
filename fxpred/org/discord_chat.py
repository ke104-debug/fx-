"""Discord でボスAIと話す (利用者の要望 2026-11-28: ディスコードにコメントするだけで返信がほしい)。
5分監視 (週末は速報の見張り) のたびに、通知チャンネル (AI_FX) の新しいコメントを読み、利用者のコメントにボスAIが返信する。
- 必要な秘密の値: DISCORD_BOT_TOKEN (GitHub の Actions secrets。ボットはチャンネルを読む・書く権限と Message Content Intent)
- チャンネルは通知用の webhook から調べる (webhook の情報にチャンネルの番号がある)
- ボット・webhook (システム自身の通知) のコメントには返信しない。1回に3件まで
- 会話は journal/boss_chat.csv に残る (公開。金額・持ち高は書かない)。ボットのトークンは表示も保存もしない"""
import json
import os
import time
import urllib.request

API = "https://discord.com/api/v10"


def _req(path: str, token: str, body: dict | None = None):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bot {token}", "Content-Type": "application/json",
                                          "User-Agent": "DiscordBot (https://github.com/ke104-debug/fx-, 1.0)"},
                                 method="POST" if body is not None else "GET")
    return json.load(urllib.request.urlopen(req, timeout=20))


def _channel_id() -> str | None:
    if os.getenv("DISCORD_CHANNEL_ID"):
        return os.getenv("DISCORD_CHANNEL_ID")
    from .. import notify
    url = notify.webhook_url()
    if not url:
        return None
    info = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "fxpred"}), timeout=15))
    return info.get("channel_id")


def check(st: dict, dry: bool = False, max_n: int = 3) -> int:
    """新しいコメントに返信する。返信した件数を返す。状態 st['discord_last'] に最後に読んだコメントの番号を残す"""
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        return 0
    ch = st.get("discord_channel") or _channel_id()
    if not ch:
        print("Discord の会話: チャンネルが分かりません"); return 0
    st["discord_channel"] = ch
    last = st.get("discord_last")
    msgs = _req(f"/channels/{ch}/messages?limit=20" + (f"&after={last}" if last else ""), token)
    msgs = sorted(msgs, key=lambda m: int(m["id"]))
    if not msgs:
        return 0
    st["discord_last"] = msgs[-1]["id"]
    if not last:
        # 初回は、15分以内のコメントだけに答える (古いコメントにまとめて返事をしない)
        cutoff = (int(time.time()) - 15 * 60) * 1000 - 1420070400000   # Discord の番号は作られた時刻を含む
        msgs = [m for m in msgs if (int(m["id"]) >> 22) >= cutoff]
    me = st.get("discord_bot_id")
    n = 0
    from . import boss_chat
    for m in msgs:
        a = m.get("author", {})
        if a.get("bot") or m.get("webhook_id") or a.get("id") == me:
            continue
        q = (m.get("content") or "").strip()
        if not q:
            if not st.get("discord_warned_intent"):
                print("Discord の会話: 本文が読めません (ボットの Message Content Intent がオフの可能性)")
                st["discord_warned_intent"] = True
            continue
        if n >= max_n:
            break
        ans = boss_chat.ask(q[:300], "利用者 (Discord)", req=f"discord-{m['id']}")
        n += 1
        print(f"Discord の会話: {q[:40]} → {ans[:80]}")
        if not dry:
            _req(f"/channels/{ch}/messages", token, {"content": ans[:1900],
                                                    "message_reference": {"message_id": m["id"], "fail_if_not_exists": False},
                                                    "allowed_mentions": {"replied_user": True}})
    return n
