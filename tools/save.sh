#!/bin/bash
# 記録をリポジトリに保存する (GitHub Actions の中で使う)。使い方: bash tools/save.sh "メッセージ" ファイルやフォルダ...
# 2026-11-27 の点検で発覚: 以前はぶつかった時に git reset --hard で、その回の記録 (見張り中の取引を含む) を捨てていた。
# → 最新を取り込んでやり直す。自動で合わせられない時は、こちらの記録を残して合わせる。記録は捨てない
msg="$1"; shift
for f in "$@"; do [ -e "$f" ] && git add "$f"; done
git diff --cached --quiet && exit 0
git commit -qm "$msg"
for i in 1 2 3 4 5; do
  if git pull --rebase --autostash -q; then
    git push -q && exit 0
  else
    git rebase --abort 2>/dev/null
    git pull --no-rebase -X ours --no-edit -q && git push -q && exit 0
  fi
  sleep $((i * 5))
done
echo "::warning::保存に失敗 (5回試した)"
echo "保存に失敗 (5回試した)"
exit 0
