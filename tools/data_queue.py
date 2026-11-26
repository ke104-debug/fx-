"""データ取得・検証の順番待ち (夜の検証部屋から1日1回)。1分足の取得が動いていなければ、次の1つを始める。
config/data_queue.json: [{"workflow": "duka-m1.yml", "inputs": {...}} ...]。GitHub Actions の中で gh を使う (無料)。"""
import json
import subprocess
import sys

Q = "config/data_queue.json"
q = json.load(open(Q))
if not q:
    print("順番待ちなし"); sys.exit()
busy = subprocess.run(["gh", "run", "list", "--workflow", "duka-m1.yml", "-L", "5", "--json", "status", "-q", ".[].status"],
                      capture_output=True, text=True).stdout
if "queued" in busy or "in_progress" in busy:
    print("1分足の取得が動いているので待つ"); sys.exit()
job = q.pop(0)
cmd = ["gh", "workflow", "run", job["workflow"], "-r", "main"] + sum((["-f", f"{k}={v}"] for k, v in job.get("inputs", {}).items()), [])
print(" ".join(cmd))
subprocess.run(cmd, check=True)
json.dump(q, open(Q, "w"), ensure_ascii=False, indent=1)
