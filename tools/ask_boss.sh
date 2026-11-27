#!/usr/bin/env bash
# 手元 (Claude Code など) からボスAIに話しかける: bash tools/ask_boss.sh "質問" [話しかけた人]
# GitHub の REST API だけを使う (Claude Code の環境では GraphQL が使えないため)
set -e
Q="$1"; A="${2:-Claude Code}"; R=ke104-debug/fx-
T=$(date -u +%Y-%m-%dT%H:%M:%SZ)
gh api "repos/$R/actions/workflows/boss-chat.yml/dispatches" -f ref=main -f "inputs[question]=$Q" -f "inputs[asker]=$A"
for i in $(seq 1 30); do id=$(gh api "repos/$R/actions/workflows/boss-chat.yml/runs?per_page=5" -q "[.workflow_runs[]|select(.created_at>=\"$T\")][0].id"); [ -n "$id" ] && [ "$id" != "null" ] && break; sleep 5; done
for i in $(seq 1 60); do [ "$(gh api repos/$R/actions/runs/$id -q .status)" = completed ] && break; sleep 5; done
gh run view "$id" -R "$R" --log 2>/dev/null | sed -E 's/^.*Z //' | sed -n '/^ボスAI (/,/^\[記録\|^ボスAIとの会話\|^\[main/p' | grep -v "^\[main\|^ボスAIとの会話" || echo "答えを読めませんでした (run $id)"
