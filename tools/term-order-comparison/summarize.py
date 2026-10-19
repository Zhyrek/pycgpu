"""
Summarize results/: phases that differ after scrambling, and median stage timings.

usage: python summarize.py [RESULTS_DIR]
"""
import glob
import json
import os
import statistics
import sys

BRANCHES = ["develop", "739", "bocklund"]
results = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


for mode in ["add", "addmul"]:
    for branch in BRANCHES:
        path = os.path.join(results, f"scramble_{mode}_{branch}.jsonl")
        if not os.path.exists(path):
            continue
        recs = rows(path)
        compared = [r for r in recs if "differ" in r]
        differ = [r for r in compared if r["differ"]]
        print(f"scramble {mode:6s} {branch:9s} compared {len(compared)}, errors {len(recs) - len(compared)}, differ {len(differ)}")
        if branch != "develop":
            for r in differ:
                print(f"    {r['db']} {r['phase']}: {r['n_out_differ']}/{r['n_out']} outputs, max rel. diff. {r['maxrel']:.1e}")

for name, keys in [("timing_stages.jsonl", ["load", "models", "compile"]), ("timing_eq.jsonl", ["equilibrium_total"])]:
    path = os.path.join(results, name)
    if not os.path.exists(path):
        continue
    recs = rows(path)
    for key in keys:
        base = statistics.median(r["t"][key] for r in recs if r["branch"] == "develop")
        line = []
        for branch in BRANCHES:
            xs = [r["t"][key] for r in recs if r["branch"] == branch]
            if xs:
                m = statistics.median(xs)
                line.append(f"{branch} {m:7.3f} s [{min(xs):.3f}-{max(xs):.3f}] ({(m / base - 1) * 100:+.1f}%)")
        print(f"{key:17s} " + "   ".join(line))
