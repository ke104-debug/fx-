---
name: fx-diagnose
description: 利用者が「FX分析して」「ドル円どう？」「相場診断」などと指示したときに、米ドル絡みの通貨ペアを分析して 買い/売り/HOLD とその理由を出す。指示されたときだけ実行する (常時稼働しない)。
---

# FX 相場診断

現在の対象は **USD/JPY のみ** (利用者の指示)。ほかのペアは明示的に頼まれた時だけ診断する。

## 手順
1. リポジトリのルートで診断材料を取得する:
   `python -m fxpred diagnose [ペア...]`
   (テクニカル: 15分/1時間/4時間足のトレンド・RSI・MACD・ボリンジャー・高値安値、AIモデルの確率、マクロ市場の動き)
2. WebSearch で最新の材料を調べる (まとめて同時に検索する):
   - 対象通貨の直近24時間のニュース (中央銀行の発言・要人発言・政治/地政学)
   - 今日〜明日の重要経済指標の予定と、直近の結果 (予想との差)
   - 金利見通し (FRB・日銀・ECB など)
3. 判断ルール:
   - テクニカル・ファンダメンタルズ (ニュース/指標/金利)・マクロ市場の方向が **そろった時だけ** BUY / SELL
   - 材料が食い違う、重要指標の発表が数時間以内、方向感がない → HOLD
   - 過去検証ではテクニカルもAIモデルも正解率は約50〜53% で優位性が未確認。AIモデルの確率を決め手にしない
4. 次の形式で回答し、Discord にも同じ形で送る (数量の目安は不要。利用者が自分で決める):
   【現在の相場】価格・24時間の値幅・相場状態 (通常/トレンド/荒れ)・急変の有無 (`fxpred.edge.anomaly`)
   【判定】買い / 売り / 様子見 / ノートレード推奨 (`fxpred.edge.no_trade` の理由があれば優先)
   【信頼度】高・中・低 と、そろった条件の数 (`fxpred.edge.confluence`。5つ以上で高確度)
   【判断の理由】【売買戦略】(入る条件・目安・損切り・利確①②・撤退ライン)【警戒事項】
   【前回シグナルの経過】【有効期限】
   以下は旧形式 (参考):
   - **診断: BUY / SELL / HOLD** (想定期間: 原則4時間)
   - 理由: テクニカル / ファンダメンタルズ / マクロ市場 をそれぞれ1〜2行、情報源の日時つき
   - 注意点: 反対方向のリスク、近い支持線・抵抗線、予定されている指標発表
5. 記録する (後で答え合わせするため、ペアごとに):
   `python -m fxpred record USD/JPY HOLD "理由の要約" 4`
6. Discord に送る (要点だけ): `discord-notify info "FX診断 <時刻>" "<ペア: 診断 - 理由の要約 を1行ずつ>"`
7. 記録をコミットして push する (journal/diagnoses.csv)。
8. Web ページ (https://claude.ai/artifact/Td8NvtQZftsB8fGzcoPE45) を更新する:
   - `python -m fxpred snapshot` の出力を technical、診断結果を ai として
     `{"updatedAt": ISO時刻, "ai": {time_utc, horizon_h, verdict, price, confidence, summary, fundamental, macro, technical, levels, events, risk, sources:[{title,url}]}, "technical": <snapshot>}` を JSON ファイルに書く
   - ArtifactData (ToolSearch で読み込む) の set で collection `state` / doc_id `latest` に file_path で書き込む
   - 同じ内容を collection `diagnoses` に doc_id `session_<セッションID>` (無ければ時刻) で `{createdAt, pairs:["USD/JPY"], horizon, result: <上のJSONの文字列>}` として書き込む
   - ArtifactData が使えないセッション (Web ページから起動された時など) では、最終回答の json ブロックで返せばページ側が書き込む
9. 最後に必ず「AIの分析であり利益を保証しない。売買の最終判断はご自身で」と添える。

## 答え合わせ
利用者に精度を聞かれたら `python -m fxpred evaluate` を実行して、正解率と件数を正直に伝える。
件数が少ない (30件未満) うちは偶然の可能性が高いことも伝える。
