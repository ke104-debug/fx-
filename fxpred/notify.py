import json
import os
import urllib.request


def webhook_url() -> str | None:
    url = os.getenv("DISCORD_WEBHOOK_URL")
    path = os.path.expanduser("~/.config/discord-notify/webhook_url")
    if not url and os.path.exists(path):
        url = open(path).read().strip()
    return url


def send(title: str, description: str, color: int = 3447003, fields: list | None = None) -> None:
    url = webhook_url()
    if not url:
        print("[notify] DISCORD_WEBHOOK_URL 未設定のため送信しません\n" + title + "\n" + description)
        return
    embed = {"title": title[:250], "description": description[:3900], "color": color}
    if fields:
        embed["fields"] = fields[:25]
    req = urllib.request.Request(url, data=json.dumps({"embeds": [embed]}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "fxpred"})
    urllib.request.urlopen(req, timeout=15).read()
