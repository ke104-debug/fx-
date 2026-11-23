import os

PAIRS = {  # Yahoo Finance のティッカー -> 表示名
    "USDJPY=X": "USD/JPY", "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD",
    "AUDUSD=X": "AUD/USD", "NZDUSD=X": "NZD/USD", "USDCAD=X": "USD/CAD",
    "USDCHF=X": "USD/CHF",
}
INTERVAL = "15m"
PERIOD = "60d"                                   # Yahoo の15分足は直近60日まで
HORIZON = int(os.getenv("FX_HORIZON", "4"))      # 何本先を予想するか (4本 = 1時間後)
THRESHOLD = float(os.getenv("FX_THRESHOLD", "0.60"))  # シグナルを出す確率のしきい値
COST_PIPS = float(os.getenv("FX_COST_PIPS", "1.0"))   # 往復スプレッド等のコスト (pips)
STATE_FILE = os.getenv("FX_STATE_FILE", "state/last_signals.json")
DATA_DIR = os.getenv("FX_DATA_DIR", "data")


def pip_size(ticker: str) -> float:
    return 0.01 if "JPY" in ticker else 0.0001
