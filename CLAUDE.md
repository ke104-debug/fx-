# fx-predictor

- 相場の分析を頼まれたら `.claude/skills/fx-diagnose/SKILL.md` の手順で診断する。常時稼働はしない (利用者が指示したときだけ)。
- 検証のとき、Yahoo の為替日足は日付が約1日ずれている (`fxpred/daily.py` 冒頭の注意参照)。未来の情報が混ざって見せかけの高精度にならないよう、時刻のそろえ方を必ず確かめる。
- 診断の記録 `journal/diagnoses.csv` は答え合わせに使うのでコミットして残す。
