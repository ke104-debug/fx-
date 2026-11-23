"""使い方:
  python -m fxpred diagnose [USD/JPY ...]          # 今の相場の診断材料を出す (指定なしで7ペア)
  python -m fxpred record PAIR BUY|SELL|HOLD "理由" [時間h=4]  # 最終診断を記録
  python -m fxpred evaluate                        # 過去の診断を答え合わせ
  python -m fxpred backtest                        # 15分足モデルの過去検証
  python -m fxpred snapshot                        # Web ページ用に USD/JPY のテクニカル予想とシステム状況を JSON で出す
  python -m fxpred strategies [コストpips]           # 売買作戦 (レンジ逆張り等) を1時間足2.8年で検証
  python -m fxpred portfolio                       # 複数作戦で100万円を運用した想定のシミュレーション
  python -m fxpred scan [--dry]                    # 相場状態を判定し、作戦のシグナルが出た時だけ Discord に通知
  python -m fxpred morning [--dry]                 # 毎朝: 昨日の振り返り・自己改善・今日の予想を Discord に送る
  python -m fxpred study                           # 日足(2004年〜)でテクニカル/マクロの効き目を検証
"""
import json
import sys

from . import backtest, data, diagnose, features


def main(argv):  # noqa: C901
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
    elif cmd == "strategies":
        from . import strategy_report
        strategy_report.run(float(args[0]) if args else 0.5)
    elif cmd == "portfolio":
        from . import portfolio
        portfolio.run()
    elif cmd == "scan":
        from . import scan
        scan.run(dry="--dry" in args)
    elif cmd == "morning":
        from . import dailyplan
        dailyplan.run(dry="--dry" in args)
    elif cmd == "testnotify":
        from . import testnotify
        testnotify.run(dry="--dry" in args)
    elif cmd == "auto":
        # GitHub Actions から毎時実行: 日本時間10時台の最初の1回だけ朝の予想・振り返り、その後に監視
        import os, json, pandas as pd
        from . import scan, dailyplan
        now = pd.Timestamp.now(tz="Asia/Tokyo")
        path = os.getenv("FX_SCAN_STATE", "state/scan.json")
        st = json.load(open(path)) if os.path.exists(path) else {}
        if now.weekday() == 5 and now.hour == 10 and st.get("weekly_done") != f"{now:%Y-%m-%d}" or "--weekly" in args:
            from . import periodic
            periodic.run(monthly=now.day <= 7 or "--monthly" in args)   # 毎月最初の土曜は月次も
            st = json.load(open(path)) if os.path.exists(path) else {}
            st["weekly_done"] = f"{now:%Y-%m-%d}"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            json.dump(st, open(path, "w"), ensure_ascii=False, indent=1)
            if now.weekday() == 5:
                return
        if now.hour == 10 and st.get("morning_done") != f"{now:%Y-%m-%d}" and now.weekday() < 5 or "--morning" in args:
            dailyplan.run()
            st = json.load(open(path)) if os.path.exists(path) else {}
            st["morning_done"] = f"{now:%Y-%m-%d}"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            json.dump(st, open(path, "w"), ensure_ascii=False, indent=1)
        scan.run()
    elif cmd == "study":
        from . import daily
        daily.run()
    else:
        print(__doc__)


main(sys.argv[1:])
