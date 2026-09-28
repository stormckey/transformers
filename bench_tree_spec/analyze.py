"""Summarize a results JSON produced by modal_app.py.

python bench_tree_spec/analyze.py bench_tree_spec/results/<file>.json
"""

import json
import math
import sys
from collections import defaultdict


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main(path):
    data = json.load(open(path))
    print(f"model={data[0]['model']} gpu={data[0]['gpu']} max_new_tokens={data[0]['max_new_tokens']}\n")
    header = (
        f"{'task':<10} {'method':<20} {'speedup':>8} {'tok/s':>7} {'tok/fwd':>8} {'accept%':>8} "
        f"{'tree%':>6} {'draft ms/fwd':>12} {'exact':>6} {'match%':>7}"
    )
    all_rows = defaultdict(list)
    for task_result in data:
        recs = task_result["records"]
        by_prompt = defaultdict(dict)
        for r in recs:
            by_prompt[r["prompt_idx"]][r["method"]] = r
        methods = list(dict.fromkeys(r["method"] for r in recs))
        print(header)
        for m in methods:
            rows = [p[m] for p in by_prompt.values()]
            speed = [p["vanilla"]["latency"] / p[m]["latency"] for p in by_prompt.values()]
            all_rows[m].extend(zip(rows, speed))
            print(fmt(task_result["task"], m, rows, speed))
        print()

    print(header)
    for m, pairs in all_rows.items():
        print(fmt("ALL", m, [p[0] for p in pairs], [p[1] for p in pairs]))


def fmt(task, m, rows, speed):
    toks = sum(r["new_tokens"] for r in rows)
    lat = sum(r["latency"] for r in rows)
    fwd = sum(r["forwards"] for r in rows)
    prop = sum(r["proposed"] for r in rows)
    acc = sum(r["accepted"] for r in rows)
    tree = sum(r["tree_steps"] for r in rows)
    draft = sum(r["draft_time"] for r in rows)
    exact = sum(r["exact_match"] for r in rows) / len(rows)
    # fraction of vanilla-identical prefix, averaged
    match = sum(r["match_prefix"] / max(r["new_tokens"], 1) for r in rows) / len(rows)
    return (
        f"{task:<10} {m:<20} {gmean(speed):>7.2f}x {toks / lat:>7.1f} {toks / fwd:>8.2f} "
        f"{(100 * acc / prop if prop else 0):>7.1f}% {100 * tree / fwd:>5.0f}% {1000 * draft / fwd:>12.2f} "
        f"{100 * exact:>5.0f}% {100 * min(match, 1):>6.0f}%"
    )


if __name__ == "__main__":
    main(sys.argv[1])
