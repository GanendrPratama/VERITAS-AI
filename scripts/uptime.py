"""Reports orchestrator availability from logs/uptime.jsonl (written by
scripts/supervise.py): the share of probes where GET /state answered within
the dashboard's 5s timeout, overall and per day, plus restarts.

Only time the supervisor was running counts -- a deliberate scripts/stop.sh
isn't downtime.

Usage: python scripts/uptime.py [--days N] [path/to/uptime.jsonl]
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

DEFAULT_LOG = Path(__file__).resolve().parent.parent / "logs" / "uptime.jsonl"
TARGET = 98.0


def summarize(lines, since=0):
    days = defaultdict(lambda: [0, 0])  # day -> [probes, ok]
    restarts = defaultdict(int)  # reason -> count
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut short by a hard kill
        if row.get("ts", 0) < since:
            continue
        day = time.strftime("%Y-%m-%d", time.localtime(row["ts"]))
        if "probes" in row:
            days[day][0] += row["probes"]
            days[day][1] += row["ok"]
        elif row.get("event") == "restart":
            restarts[row.get("reason", "exited")] += 1
    return dict(days), dict(restarts)


def pct(probes, ok):
    return 100.0 * ok / probes if probes else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=DEFAULT_LOG, type=Path)
    ap.add_argument("--days", type=float, default=None, help="only the last N days")
    args = ap.parse_args()
    if not args.log.is_file():
        print(f"no uptime log at {args.log} -- start the system with scripts/start.sh or start.ps1 first")
        return
    since = time.time() - args.days * 86400 if args.days else 0
    days, restarts = summarize(args.log.read_text().splitlines(), since)
    total = [sum(d[0] for d in days.values()), sum(d[1] for d in days.values())]
    for day in sorted(days):
        print(f"{day}  {pct(*days[day]):6.2f}%  ({days[day][1]}/{days[day][0]} probes ok)")
    overall = pct(*total)
    if overall is None:
        print("no probes recorded yet")
        return
    print(f"overall     {overall:6.2f}%  target {TARGET}% -- {'MET' if overall >= TARGET else 'MISSED'}")
    for reason, n in sorted(restarts.items(), key=lambda kv: -kv[1]):
        print(f"restarts: {n} x {reason}")


if __name__ == "__main__":
    main()
