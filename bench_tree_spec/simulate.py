"""Offline replay of drafting policies against recorded greedy outputs.

With greedy decoding the target output does not depend on the drafts, so given the vanilla output we can replay
generation exactly: at every step the policy proposes a draft tree from the tokens seen so far, the accepted length is
the longest root-to-node path that matches the true continuation, and one target forward is spent. This measures
tokens per forward (and draft sizes / draft CPU time) for any policy at zero GPU cost.

    python bench_tree_spec/simulate.py bench_tree_spec/results/dev_vanilla.json
    python bench_tree_spec/simulate.py bench_tree_spec/results/dev_vanilla.json --cache

`--cache` replays the requests of each task in order through one `SuffixDecodingCache`, so later requests can draft
from earlier responses, and compares it with per-request (local-only) drafting.
"""

import json
import sys
import time
from collections import defaultdict

from transformers.generation.suffix_tree import SuffixDecodingCache, SuffixTree


def replay(rows, policy, max_depth=64):
    stats = defaultdict(float)
    per_row = []
    for row in rows:
        prompt, out = row["prompt"], row["output"]
        tree = SuffixTree(max_depth=max_depth)
        tree.extend(0, prompt)
        seq = list(prompt)
        pos = fwd = 0
        sizes = []
        while pos < len(out):
            t0 = time.perf_counter()
            token_ids, parents = policy(tree, seq[-max_depth:])
            stats["draft_s"] += time.perf_counter() - t0
            children = {}
            for i, (tok, par) in enumerate(zip(token_ids, parents)):
                children.setdefault((par, tok), i)
            node, k = -1, 0
            while pos + k < len(out) and (node := children.get((node, out[pos + k]))) is not None:
                k += 1
            new = out[pos : pos + k + 1]
            tree.extend(0, new)
            seq.extend(new)
            pos += len(new)
            fwd += 1
            sizes.append(len(token_ids))
        stats["tokens"] += len(out)
        stats["forwards"] += fwd
        stats["draft_nodes"] += sum(sizes)
        per_row.append({"tokens": len(out), "forwards": fwd, "sizes": sizes})
    return stats, per_row


# ---------------------------------------------------------------- policies


def linear(n=32, min_prob=0.1, factor=3.0, offset=5.0):
    def policy(tree, ctx):
        d = tree.speculate(ctx, n, min_prob, max_spec_factor=factor, max_spec_offset=offset)
        return d.token_ids, d.parents

    return policy


def tree_best(n=64, branch=4, min_prob=0.1, factor=3.0, offset=5.0):
    def policy(tree, ctx):
        d = tree.speculate_tree(ctx, n, min_prob, branch, max_spec_factor=factor, max_spec_offset=offset)
        return d.token_ids, d.parents

    return policy


def tree_merged(n=64, branch=4, min_prob=0.1, factor=3.0, offset=5.0, max_matches=8):
    """Union of the drafts of the `max_matches` longest matching suffixes, then keep the `n` most likely nodes."""

    def policy(tree, ctx):
        matches = []
        for match_len in range(1, len(ctx)):
            res = tree.match_context(ctx[-match_len:])
            if res is None:
                break
            matches.append((match_len, res))
        drafts = []
        for match_len, (node, idx) in reversed(matches):
            cap = int(match_len * factor + offset + 1e-6)
            d = tree.speculate_tree_path(node, idx, min(n, cap), min_prob, branch)
            if d.token_ids:
                drafts.append(d)
            if len(drafts) == max_matches:
                break
        trie = {}  # (parent_trie_id, token) -> [trie_id, prob, depth]
        for d in drafts:
            ids = []
            for tok, par, prob, depth in zip(d.token_ids, d.parents, d.probs, d.depths):
                key = (ids[par] if par >= 0 else -1, tok)
                entry = trie.get(key)
                if entry is None:
                    entry = trie[key] = [len(trie), prob, depth]
                elif prob > entry[1]:
                    entry[1] = prob
                ids.append(entry[0])
        # parents always have prob >= children, so a (prob desc, depth asc) order is topological
        chosen = sorted(trie.items(), key=lambda kv: (-kv[1][1], kv[1][2]))[:n]
        remap = {}
        token_ids, parents = [], []
        for (par, tok), (tid, _, _) in chosen:
            if par != -1 and par not in remap:
                continue
            remap[tid] = len(token_ids)
            token_ids.append(tok)
            parents.append(remap[par] if par != -1 else -1)
        return token_ids, parents

    return policy


POLICIES = {
    "linear_n32": linear(),
    "tree_b4_n64": tree_best(),
    "tree_b4_n64_cap_x6_o16": tree_best(factor=6, offset=16),
    "tree_b4_n64_nocap": tree_best(factor=1e9, offset=1e9),
    "tree_b4_n64_p05_nocap": tree_best(min_prob=0.05, factor=1e9, offset=1e9),
    "tree_b8_n64_p05_nocap": tree_best(branch=8, min_prob=0.05, factor=1e9, offset=1e9),
    "merged_b4_n64_nocap": tree_merged(factor=1e9, offset=1e9),
    "merged_b4_n64_p05_nocap": tree_merged(min_prob=0.05, factor=1e9, offset=1e9),
    "merged_b4_n64_p05_nocap_m2": tree_merged(min_prob=0.05, factor=1e9, offset=1e9, max_matches=2),
    "merged_b8_n128_p02_nocap": tree_merged(n=128, branch=8, min_prob=0.02, factor=1e9, offset=1e9),
    "tree_b8_n128_p02_nocap": tree_best(n=128, branch=8, min_prob=0.02, factor=1e9, offset=1e9),
    "tree_b8_n128_p01_nocap": tree_best(n=128, branch=8, min_prob=0.01, factor=1e9, offset=1e9),
    "tree_b16_n256_p01_nocap": tree_best(n=256, branch=16, min_prob=0.01, factor=1e9, offset=1e9),
    "tree_b8_n256_p005_nocap": tree_best(n=256, branch=8, min_prob=0.005, factor=1e9, offset=1e9),
    "tree_b8_n128_p02_depth32": tree_best(n=128, branch=8, min_prob=0.02, factor=1e9, offset=1e9),
}


def accepted_length(token_ids, parents, out, pos):
    children = {}
    for i, (tok, par) in enumerate(zip(token_ids, parents)):
        children.setdefault((par, tok), i)
    node, k = -1, 0
    while pos + k < len(out) and (node := children.get((node, out[pos + k]))) is not None:
        k += 1
    return k


LINEAR_KWARGS = {"max_spec_tokens": 32, "min_token_prob": 0.1}
TREE_KWARGS = {
    "max_spec_tokens": 128,
    "min_token_prob": 0.02,
    "branch_factor": 8,
    "max_spec_factor": 1e9,
    "max_spec_offset": 1e9,
}


def replay_cache(rows, tree_drafts, source_mode, max_cached_requests=64, max_depth=64):
    """Tokens per forward and draft nodes per forward for a stream of requests sharing one cache."""
    cache = SuffixDecodingCache(max_depth, max_cached_requests, source_mode)
    tokens = forwards = nodes = 0
    for row in rows:
        prompt, out = row["prompt"], row["output"]
        req_id = cache.start_request(prompt)
        seq = list(prompt)
        pos = 0
        while pos < len(out):
            if tree_drafts:
                draft = cache.speculate_tree(req_id, seq[-max_depth:], **TREE_KWARGS)
            else:
                draft = cache.speculate(req_id, seq[-max_depth:], **LINEAR_KWARGS)
            k = accepted_length(draft.token_ids, draft.parents, out, pos)
            new = out[pos : pos + k + 1]
            cache.add_active_response(req_id, new)
            seq.extend(new)
            pos += len(new)
            forwards += 1
            nodes += len(draft.token_ids)
        cache.stop_request(req_id)
        tokens += len(out)
    return tokens / forwards, nodes / forwards


def main_cache(path):
    data = json.load(open(path))
    columns = [
        ("suffix local", False, "local_only"),
        ("suffix +shared", False, "local_and_global"),
        ("tree local", True, "local_only"),
        ("tree +shared", True, "local_and_global"),
    ]
    print(f"{'task':<12}" + "".join(f"{name:>20}" for name, _, _ in columns) + "   (tokens/forward, draft nodes)")
    for d in data:
        cells = []
        for _, tree_drafts, mode in columns:
            tpf, q = replay_cache(d["rows"], tree_drafts, mode)
            cells.append(f"{tpf:6.2f} q{q:5.1f}")
        print(f"{d['task']:<12}" + "".join(f"{c:>20}" for c in cells), flush=True)


def main(path, names=None):
    data = json.load(open(path))
    names = names or list(POLICIES)
    print(f"{'policy':<28} " + " ".join(f"{d['task']:>18}" for d in data) + f" {'draft ms/step':>14}")
    for name in names:
        cells, draft_s, steps = [], 0.0, 0
        for d in data:
            st, _ = replay(d["rows"], POLICIES[name])
            cells.append(f"{st['tokens'] / st['forwards']:6.3f} q{st['draft_nodes'] / st['forwards']:5.1f}")
            draft_s += st["draft_s"]
            steps += st["forwards"]
        print(f"{name:<28} " + " ".join(f"{c:>18}" for c in cells) + f" {1000 * draft_s / steps:>14.2f}", flush=True)


if __name__ == "__main__":
    if "--cache" in sys.argv:
        main_cache(sys.argv[1])
    else:
        main(sys.argv[1], sys.argv[2:] or None)
