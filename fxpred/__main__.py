"""使い方:
  python -m fxpred diagnose [USD/JPY ...]          # 今の相場の診断材料を出す (指定なしで7ペア)
  python -m fxpred record PAIR BUY|SELL|HOLD "理由" [時間h=4]  # 最終診断を記録
  python -m fxpred evaluate                        # 過去の診断を答え合わせ
  python -m fxpred backtest                        # 15分足モデルの過去検証
  python -m fxpred snapshot                        # Web ページ用に USD/JPY のテクニカル予想とシステム状況を JSON で出す
  python -m fxpred study                           # 日足(2004年〜)でテクニカル/マクロの効き目を検証
"""
import json
import sys

from . import backtest, data, diagnose, features


def main(argv):
    cmd, args = (argv[0] if argv else ""), argv[1:]
    if cmd == "diagnose":
        print(json.dumps(diagnose.run(args or None), ensure_ascii=False, indent=1))
    elif cmd == "record":
        pair, verdict, reason = args[0], args[1], args[2]
        h = float(args[3]) if len(args) > 3 else 4
        print(json.dumps(diagnose.record(pair, verdict, reason, h), ensure_ascii=False))
    elif cmd == "evaluate":
        print(diagnose.evaluate())
    elif cmd == "backtest":
        print(backtest.report(backtest.walk_forward(features.build_all(data.fetch()))))
    elif cmd == "snapshot":
        from . import snapshot
        print(json.dumps(snapshot.run(), ensure_ascii=False, indent=1))
    elif cmd == "study":
        from . import daily
        daily.run()
    else:
        print(__doc__)


main(sys.argv[1:])
