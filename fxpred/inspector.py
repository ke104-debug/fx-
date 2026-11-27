"""点検部隊 (利用者の指示 2026-11-27: システムのミスを徹底的に探し、同じミスを2度とさせない)。
監査AI (考え方の問題点を挙げる) とは別。こちらは「決めたとおりに動いているか」を機械的に毎回確かめる。

- 毎時 (light): 送った合図と作戦の条件の一致・損切りと利確の向き・見張り中の取引・記録の新しさ・影の運用が記録されているか・
  通知している作戦の根拠の強さ・記録ファイルの形
- 毎晩 (deep, 夜の検証部屋): 上に加えて、GitHub Actions の24時間分のログのエラー・5分監視の途切れ・
  直近24時間のコードの変更を AI (Groq の無料枠、だめなら Gemini) に読ませて食い違いを探させる
- 見つかった問題は reports/inspector.md と journal/inspector.csv に記録し、新しい問題は AI_FX に1日1回知らせる
- 過去のミスは config/known_mistakes.json に登録し、その点検を毎回すべて流す (同じミスの再発を止める)"""
import csv
import json
import os
import subprocess

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
J = lambda *p: os.path.join(ROOT, *p)
LOG = J("journal", "inspector.csv")
REPORT = J("reports", "inspector.md")
KNOWN = J("config", "known_mistakes.json")
STATE = os.getenv("FX_INSPECT_STATE", "state/inspector.json")


def _now():
    return pd.Timestamp.now(tz="UTC")


def _market_open(t=None) -> bool:
    j = (t or _now()) + pd.Timedelta(hours=9)
    return not (j.weekday() == 5 and j.hour >= 7 or j.weekday() == 6 or j.weekday() == 0 and j.hour < 7)


# ---------------------------------------------------------------- 点検 (どれも (問題なし?, 内容) を返す)
def c_signal_rules(h):
    """送った合図が、作戦の名前どおりの条件を満たしていたか (M001)"""
    from . import signal_audit
    sg = pd.read_csv(J("journal", "signals.csv"))
    sg = sg[(sg.sent.astype(str) == "True")]
    bad = []
    for r in sg.itertuples():
        t = pd.Timestamp(r.time_utc, tz="UTC")
        if t < h.index[0] or t < pd.Timestamp("2026-11-27 09:00", tz="UTC"):   # 修正前の2回は記録済み (CHANGELOG)
            continue
        j = h.index.searchsorted(t, side="right") - 1
        ok, why = signal_audit.rule_ok(r.key, int(r.side), h, j)
        if not ok:
            bad.append(f"{r.time_utc} {r.key}: {why}")
    return not bad, " / ".join(bad) or "送った合図はすべて作戦の条件どおり"


def c_signal_geometry(h):
    """損切り・利確が正しい側にあるか、入る値段がその時の値段から離れすぎていないか"""
    sg = pd.read_csv(J("journal", "signals.csv")).tail(60)
    bad = []
    for r in sg.itertuples():
        s = int(r.side)
        if s * (r.entry - r.stop) <= 0:
            bad.append(f"{r.time_utc} 損切りが逆側 ({r.entry}→{r.stop})")
        if pd.notna(r.target) and s * (r.target - r.entry) <= 0:
            bad.append(f"{r.time_utc} 利確が逆側 ({r.entry}→{r.target})")
        t = pd.Timestamp(r.time_utc, tz="UTC")
        if h.index[0] <= t:
            px = h.Close.asof(t)
            if pd.notna(px) and abs(px - r.entry) > 0.5:
                bad.append(f"{r.time_utc} 入る値段 {r.entry} がその時の値段 {px:.3f} から50pips以上離れている")
    return not bad, " / ".join(bad[:5]) or "損切り・利確・入る値段の向きはすべて正しい"


def c_positions(h):
    from . import positions
    bad = []
    now = int(_now().timestamp())
    for p in positions.load():
        if p["side"] * (p["entry"] - p["stop"]) < 0 and p.get("key") != "U":
            bad.append(f"{p['id']} 損切りが逆側")
        if p.get("expire") and now > p["expire"] + 3 * 3600:
            bad.append(f"{p['id']} 期限を3時間以上過ぎても見張りが残っている")
    return not bad, " / ".join(bad) or "見張り中の取引に異常なし"


def c_freshness(h):
    """記録が止まっていないか (市場が開いている時だけ)"""
    if not _market_open():
        return True, "週末のため確認しない"
    lim = {"org.csv": 3, "quick_ai.csv": 1, "research.csv": 4}
    from .org import mode
    if mode.paused("5分監視の見立て"):
        lim.pop("quick_ai.csv")      # 統廃合で休止中
    bad = []
    for f, hours in lim.items():
        try:
            d = pd.read_csv(J("journal", f), usecols=["time_utc"])
            last = pd.to_datetime(d.time_utc, utc=True).max()
            if _now() - last > pd.Timedelta(hours=hours):
                bad.append(f"{f} が {(_now() - last).total_seconds() / 3600:.1f}時間 更新されていない")
        except Exception as e:
            bad.append(f"{f} を読めない ({str(e)[:40]})")
    return not bad, " / ".join(bad) or "記録は止まっていない"


def c_shadow_alive(h):
    """影の運用が本当に記録されているか (M002)"""
    from . import shadow, lab
    d = lab.prep(h.iloc[-900:])
    if "trend_up" not in d:
        d["trend_up"] = d.Close.ewm(span=480).mean().diff(24) > 0
    have = set()
    if os.path.exists(shadow.PATH):
        x = pd.read_csv(shadow.PATH)
        have = set(zip(x.strategy, x.time_utc))
    miss = []
    for name, c in shadow.CANDIDATES.items():
        t = c["fn"](d, **c["params"])
        for r in (t[t.time >= shadow.START].itertuples() if len(t) else []):
            if (name, f"{r.time:%Y-%m-%d %H:%M}") not in have:
                miss.append(f"{name} {r.time:%m/%d %H:%M}")
    return not miss, ("記録漏れ: " + " / ".join(miss[:5])) if miss else "影の運用は記録されている"


def c_live_strength(h):
    """通知している作戦の23年の根拠 (2016年以降 +0.02R 以下なら、根拠が弱いと知らせ続ける)"""
    from . import scan
    st = json.load(open(J("config", "strategy_stats.json")))["stats"]
    live = sorted(scan.notify_keys())
    weak = [f"{k} ({scan.LABEL.get(k, k)}: 23年 {st[k]['R']:+.3f}R / 2016年以降 {st[k]['R16']:+.3f}R)"
            for k in live if k in st and st[k]["R16"] <= 0.02]
    return not weak, ("根拠の弱い作戦を通知中: " + " / ".join(weak)) if weak else "通知している作戦はどれも検証で根拠あり"


def c_csv_shape(h):
    """記録ファイルの列数の食い違い (M007)"""
    bad = []
    for f in sorted(os.listdir(J("journal"))):
        if not f.endswith(".csv") or f.startswith("trades"):
            continue
        with open(J("journal", f), newline="") as fh:
            rows = list(csv.reader(fh))
        if not rows:
            continue
        n = len(rows[0])
        wrong = sum(1 for r in rows[1:] if len(r) != n)
        if wrong:
            bad.append(f"{f}: {wrong}行の列数が見出しと違う")
    return not bad, " / ".join(bad) or "記録ファイルの形は正しい"


def c_roles(h):
    """役割の点検 (K003): 判断を決める・止める部品が、根拠のない役割違いのまま使われていないか (config/roles.json)。
    あわせて、テクニカルが方向の票を持っていないか"""
    import json as _j
    bad = []
    try:
        r = _j.load(open(J("config", "roles.json")))["parts"]
        for p in r:
            if not p.get("ok") and any(x in p["role"] for x in ("決める", "止める")) and "使っていない" not in p.get("fix", ""):
                bad.append(f"{p['name']} ({p['role']}): {p.get('fix', '')[:50]}")
    except Exception as e:
        bad.append(f"役割の一覧を読めない ({str(e)[:40]})")
    from .org.meeting import TECH_SECTIONS, weights as _w
    tech = [k for k, v in _w().items() if v and k.split("|")[0] in TECH_SECTIONS]
    if tech:
        bad.append("テクニカルが方向の票の重みを持っている (会議では0にしている): " + " / ".join(tech[:3]))
    return not bad, " / ".join(bad) or "判断に使う部品はすべて根拠のある役割"


def c_news(h):
    """ニュースの読み取りが動いているか (M016: 2日間ほぼ0回だったのに気づかなかった)。直近24時間の成功の割合"""
    if not _market_open():
        return True, "週末のため確認しない"
    try:
        g = pd.read_csv(J("journal", "gemini_usage.csv"))
        g = g[(g.job == "ニュース") & (pd.to_datetime(g.time_utc, utc=True) >= _now() - pd.Timedelta(hours=24))]
    except Exception as e:
        return False, f"記録を読めない ({str(e)[:40]})"
    ok = int((g.outcome == "ok").sum())
    if len(g) >= 4 and ok / len(g) < 0.5:
        return False, f"ニュースの読み取りが24時間で {ok}/{len(g)}回しか成功していない (Gemini の混雑など)"
    if ok == 0:
        return False, "ニュースの読み取りが24時間で1回も成功していない"
    return True, f"ニュースの読み取り: 24時間で {ok}/{len(g)}回成功"


def c_live_parts(h):
    """通知している作戦の部品が、監視と同じ環境 (検証用の部品 numba なし) で動くか (M018: numba が無く流れ×タイミングが一度も動いていなかった)。
    15分足を実際に読んで合図の計算を最後まで通す"""
    import subprocess as _sp, sys as _s
    code = ("import builtins as b; r=b.__import__\n"
            "def f(n,*a,**k):\n  if n.startswith('numba'): raise ImportError(n)\n  return r(n,*a,**k)\n"
            "b.__import__=f\nfrom fxpred import live_tl, live_sf, live_flow, exit_alert, newswatch, league, explore as ex, flowtiming as ft\n"
            "d=ex.bars(tf=None, src=live_tl._bars15()); d=d[d.atr>0]; print(int(ft.trendline_signal(d)[-1]), len(d))")
    try:
        p = _sp.run([_s.executable, "-c", code], capture_output=True, text=True, timeout=180, cwd=J())
    except Exception as e:
        return False, f"部品の確認が終わらない ({str(e)[:40]})"
    if p.returncode:
        return False, "通知している作戦の部品が動かない: " + (p.stderr.strip().splitlines() or ["?"])[-1][:120]
    return True, f"通知している作戦の部品は監視と同じ環境で動く (15分足 {p.stdout.split()[-1]}本)"


LIGHT = {"作戦の部品が動くか": c_live_parts, "合図と作戦の条件": c_signal_rules, "損切り・利確の向き": c_signal_geometry, "見張り中の取引": c_positions,
         "記録の新しさ": c_freshness, "影の運用の記録": c_shadow_alive, "通知している作戦の根拠": c_live_strength,
         "記録ファイルの形": c_csv_shape, "部品の役割": c_roles, "ニュースの読み取り": c_news}


# ---------------------------------------------------------------- 毎晩だけ (GitHub のログ・AI)
def _gh(*args) -> str:
    try:
        return subprocess.run(["gh", *args], capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return ""


def c_monitor_gaps(h):
    """5分監視の途切れ (24時間、市場が開いている間に20分以上の空き)"""
    out = _gh("run", "list", "--workflow", "fx-quick.yml", "-L", "100", "--json", "createdAt,updatedAt,conclusion")
    if not out:
        return True, "GitHub に聞けないため確認できず"
    runs = [r for r in json.loads(out) if r["conclusion"] == "success"]
    spans = sorted((pd.Timestamp(r["createdAt"]), pd.Timestamp(r["updatedAt"])) for r in runs)
    since = _now() - pd.Timedelta(hours=24)
    gaps, end = [], None
    for a, b in spans:
        if b < since:
            continue
        if end is not None and a - end > pd.Timedelta(minutes=20) and _market_open(end):
            gaps.append(f"{(end + pd.Timedelta(hours=9)):%m/%d %H:%M}〜{(a + pd.Timedelta(hours=9)):%H:%M}")
        end = max(end, b) if end is not None else b
    return not gaps, ("5分監視の途切れ: " + " / ".join(gaps[:6])) if gaps else "5分監視は途切れていない"


def c_log_errors(h):
    """24時間分の実行ログのエラー (Traceback と『〜のエラー』)"""
    out = _gh("run", "list", "-L", "60", "--json", "databaseId,workflowName,createdAt,conclusion")
    if not out:
        return True, "GitHub に聞けないため確認できず"
    since = _now() - pd.Timedelta(hours=24)
    errs, failed = {}, []
    for r in json.loads(out):
        if pd.Timestamp(r["createdAt"]) < since or r["conclusion"] in ("cancelled", "skipped", ""):
            continue
        if r["conclusion"] == "failure":
            failed.append(f"{r['workflowName']} {pd.Timestamp(r['createdAt']) + pd.Timedelta(hours=9):%m/%d %H:%M}")
        log = _gh("run", "view", str(r["databaseId"]), "--log")
        for line in log.splitlines():
            msg = line.split("Z ", 1)[-1].strip()
            if any(w in msg for w in ("のエラー", "Traceback", "Error:", "できませんでした", "呼べませんでした", "保存に失敗", "見送り (失敗")) \
                    and "echo" not in msg and "[36;1m" not in msg:
                k = msg[:90]
                errs[k] = errs.get(k, 0) + 1
    items = [f"{k} ({v}回)" for k, v in sorted(errs.items(), key=lambda x: -x[1])]
    ok = not items and not failed
    txt = ("失敗した実行: " + " / ".join(failed[:5]) + " | " if failed else "") + (" / ".join(items[:8]) if items else "")
    return ok, txt or "24時間のログにエラーなし"


def c_ai_code_review(h):
    """直近24時間のコードの変更を AI に読ませて、決まりや作戦の条件との食い違いを探させる (補助。直すのは人と Claude)"""
    try:
        diff = subprocess.run(["git", "log", "--since=24 hours ago", "-p", "--no-color", "-U2", "--", "fxpred/*.py", "fxpred/org/*.py"],
                              capture_output=True, text=True, cwd=ROOT, timeout=60).stdout
    except Exception as e:
        return False, f"コードの変更を読めなかった ({type(e).__name__})"
    if not diff.strip():
        return True, "直近24時間のコードの変更なし"
    rules = ("作戦の名前どおりの条件で合図を出す (『流れの方向』なら大きな流れと同じ向き)。損切りは買いなら下・売りなら上。"
             "記録は journal/ に保存。未来の足を使わない。エラーを握りつぶしたら必ず print で残す")
    from .org import groq
    r, who = groq.ask("点検の補助",
                      "あなたは FX の通知システムのコードを点検する係。次の変更 (git diff) に、"
                      "①作戦の名前や説明と中身の食い違い ②損切り・利確・売り買いの向きの間違い ③例外を握りつぶして処理が止まっても気づけない所"
                      " ④記録が保存されない所 ⑤未来の情報が混ざる所 ⑥テクニカル指標や AI の予想を、売買の向きを決める主な根拠にしている所"
                      " (テクニカルは入る時・降りる時の調整役で、向きは検証済みの作戦と流れで決める) があれば挙げる。推測ではなく diff の中の根拠 (ファイル名と内容) を書く。"
                      '無ければ空。JSON だけ: {"problems": [{"file": "...", "what": "60字以内", "why": "60字以内"}]}\n\n'
                      f"決まり:\n{rules}\n\n変更:\n{diff[:6000]}")
    if not isinstance(r, dict):
        return False, "AI を呼べず、コードの変更の点検ができなかった (Groq・Gemini とも)"
    probs = r.get("problems", [])
    probs = [p for p in probs if isinstance(p, dict) and p.get("what")]
    txt = " / ".join(f"{p.get('file', '')}: {p['what']} ({p.get('why', '')})" for p in probs[:5])
    return not probs, (f"AI ({who}) の指摘: " + txt) if probs else f"AI ({who}) の点検で指摘なし"


DEEP = {"5分監視の途切れ": c_monitor_gaps, "実行ログのエラー": c_log_errors, "コードの変更の点検 (AI)": c_ai_code_review}


def known_checks() -> list[dict]:
    try:
        return json.load(open(KNOWN))
    except Exception:
        return []


def run(h, deep=False, dry=False) -> list[tuple[str, str, str]]:
    """全部の点検を流し、結果を記録。新しい問題 (その日まだ知らせていないもの) を通知用に返す"""
    checks = dict(LIGHT, **(DEEP if deep else {}))
    rows, msgs = [], []
    for name, fn in checks.items():
        try:
            ok, detail = fn(h)
        except Exception as e:
            ok, detail = False, f"点検そのものが失敗: {type(e).__name__} {str(e)[:80]}"
        rows.append({"time_utc": f"{_now():%Y-%m-%d %H:%M}", "check": name, "ok": ok, "detail": detail[:400], "deep": deep})
    km = known_checks()
    lines = [f"# 点検部隊 ({(_now() + pd.Timedelta(hours=9)):%Y-%m-%d %H:%M} 日本時間・{'毎晩の詳しい点検' if deep else '毎時の点検'})", ""]
    lines += [f"- {'✅' if r['ok'] else '❌'} **{r['check']}**: {r['detail']}" for r in rows]
    lines += ["", "## 過去のミスと再発防止 (config/known_mistakes.json)", ""]
    lines += [f"- {m['id']} {m['date']} {m['what']} → {m['prevent']}" for m in km]
    if not dry:
        os.makedirs(os.path.dirname(REPORT), exist_ok=True)
        open(REPORT, "w").write("\n".join(lines) + "\n")
        old = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame()
        pd.concat([old, pd.DataFrame(rows)]).tail(5000).to_csv(LOG, index=False)
    try:
        st = json.load(open(STATE))
    except Exception:
        st = {}
    # 未解決の問題の台帳とボスAIの督促 (2026-11-27 利用者の指摘 K006: NG を『よくあること』と流させない)
    from .org import boss
    try:
        b_rows, b_msgs = boss.discipline(dry=dry)
        for r in b_rows:
            r.update(time_utc=f"{_now():%Y-%m-%d %H:%M}", deep=deep)
            print(f"(ボスAI) {'OK' if r['ok'] else 'NG'} {r['check']}: {r['detail'][:150]}")
        rows += b_rows
        msgs += b_msgs
    except Exception as e:
        print("ボスAIの引き締めのエラー:", e)
        rows.append({"check": "ボス: 引き締め", "ok": False, "detail": f"動かなかった ({str(e)[:60]})", "time_utc": f"{_now():%Y-%m-%d %H:%M}", "deep": deep})
    msgs += boss.track_issues(rows, deep=deep, dry=dry)
    if not dry:
        os.makedirs(os.path.dirname(STATE) or ".", exist_ok=True)
        json.dump(st, open(STATE, "w"), ensure_ascii=False)
    for r in rows:
        print(f"(点検) {'OK' if r['ok'] else 'NG'} {r['check']}: {r['detail'][:150]}")
    return msgs
