#!/usr/bin/env bash
# 手元 (Claude Code など) からボスAIに話しかける: bash tools/ask_boss.sh "質問" [話しかけた人]
set -e
Q="$1"; A="${2:-Claude Code}"
T=$(date -u +%Y-%m-%dT%H:%M:%S)
gh workflow run boss-chat.yml -f question="$Q" -f asker="$A" >/dev/null
for i in $(seq 1 20); do id=$(gh run list --workflow boss-chat.yml -L 5 --json databaseId,createdAt -q "[.[]|select(.createdAt>=\"$T\")][0].databaseId"); [ -n "$id" ] && [ "$id" != "null" ] && break; sleep 5; done
gh run watch "$id" --exit-status >/dev/null 2>&1 || true
gh run view "$id" --log 2>/dev/null | sed -E 's/^.*Z //' | sed -n '/^ボスAI (/,/^\[/p' | grep -v '^\[' || gh run view "$id"
