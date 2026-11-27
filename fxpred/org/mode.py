"""組織の形の切り替え (2026-11-27 統廃合)。config/org_mode.json の mode が "simple" なら統廃合後、"full" なら元の組織。
元に戻す時は mode を "full" にするだけ (コードは消していない)。

統廃合後 (simple) の6部門:
  ボスAI (最終決定・決まりだけで判断) / 売買判断部 (作戦部隊 + 決済・ポジション管理) / リスク管理部 (リスク管理 + リスク統括 + 資金管理 + データ監視) /
  監査・点検部 (点検部隊 + 監査AI、組織から独立) / 情報部 (情報解析 + 相場分析。票を持たず報告の環境メモ) / 研究部 (検証 + 改善研究 + 影の運用 + 夜の検証)
休止 (simple の時): 方向の会議を判断と報告に使うこと (採点のための記録は続ける)・統括補佐AI・各部隊の相談役・5分監視の AI の見立て (Gemini の15分の見立てを含む)・反対意見のAI"""
import json
import os

PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "org_mode.json")
PAUSED = ("統括補佐", "相談役", "5分監視の見立て", "反対意見", "方向の会議の表示")


def simple() -> bool:
    try:
        return json.load(open(PATH)).get("mode", "full") == "simple"
    except Exception:
        return False


def paused(name: str) -> bool:
    return simple() and name in PAUSED
