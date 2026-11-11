# Tree-spec vs suffix decoding — Qwen3-8B, H100

Setup: bf16, SDPA, greedy, batch size 1, 30 test prompts per task (same prompts in every run), max 384 new tokens.
Every prompt runs all methods back to back in one container with the method order rotated per prompt. The baseline
("vanilla") is Hugging Face `generate` with its defaults: eager PyTorch and a dynamic KV cache. See
[Baseline sensitivity](#baseline-sensitivity-compiled-vanilla-decoding) for what a compiled baseline changes.

Three numbers are reported per method, because they answer different questions:

- **speedup**: geometric mean over prompts of vanilla latency / method latency. Each prompt counts once. Method-vs-method
  numbers are paired per prompt with a 95% bootstrap CI.
- **total time**: sum of vanilla latencies / sum of method latencies. Long generations count more; this is the
  throughput ratio of the whole prompt set.
- **tokens/forward**: generated tokens per target forward pass. It does not depend on hardware or on the baseline.

## Per-request drafting (`results/qwen3-8b-h100-v3.json`)

Speedup (geometric mean):

| task | PLD | suffix (linear) | tree, old policy (run 1) | **tree_new** | tree_new vs suffix |
|---|---|---|---|---|---|
| code_edit (CanItEdit) | 3.90x | 6.64x | 7.28x | **9.04x** | 1.36x |
| humaneval | 1.69x | 1.85x | 1.97x | **1.93x** | 1.05x |
| gsm8k | 1.41x | 1.43x | 1.52x | **1.57x** | 1.10x |
| chat (MT-Bench) | 1.11x | 1.15x | 1.18x | **1.13x** | 0.98x |
| all | 1.79x | 2.12x | 2.25x | **2.36x** | **1.11x [1.08, 1.15]** |
| all except code_edit | 1.38x | 1.45x | | **1.51x** | |

Total time and tokens/forward:

| task | suffix total time | tree_new total time | suffix tokens/forward | tree_new tokens/forward |
|---|---|---|---|---|
| code_edit | 5.28x | 6.48x | 6.21 | 7.68 |
| humaneval | 1.81x | 1.90x | 2.05 | 2.19 |
| gsm8k | 1.44x | 1.59x | 1.63 | 1.83 |
| chat | 1.14x | 1.12x | 1.26 | 1.32 |

`tree_new` = `tree_spec_decoding_num_tokens=128, branch_factor=8, min_prob=0.02, max_spec_factor=max_spec_offset=1e9`.
The tree_new/suffix ratio reproduced across three independent runs (v2: 1.112x, v3: 1.113x, v4: 1.113x).

How to read the code_edit row: per-prompt tree_new speedups range from 1.5x to 34x (median 12.9x). They are above 12x
on the 16 prompts with long copied spans (15 or more tokens accepted per forward) and between 1.5x and 8x on the
other 14. The geometric mean (9.04x) is therefore higher than the total-time ratio (6.48x), which is dominated by
the slow prompts. On chat, 8 of 30 prompts are slower than vanilla
decoding (worst 0.50x): a rejected draft still pays the extra cost of a multi-token step.

## Cross-request cache (`results/qwen3-8b-h100-v4.json`)

`SuffixDecodingCache` lets a request draft from the responses to earlier requests. Here the 30 prompts of a task
arrive one after another through one cache, so prompt *i* can use the responses to prompts 0 .. *i*-1.

| task | suffix | suffix + cache | tree_new | tree_new + cache | tree: cache vs no cache |
|---|---|---|---|---|---|
| code_edit | 6.91x | 6.92x | 9.43x | 9.15x | 0.97x [0.90, 1.03] |
| humaneval | 1.87x | 2.04x | 1.97x | 2.47x | 1.25x [1.19, 1.33] |
| gsm8k | 1.42x | 1.60x | 1.56x | 2.07x | 1.33x [1.27, 1.39] |
| chat | 1.15x | 1.09x | 1.13x | 1.17x | 1.04x [0.99, 1.09] |
| all | 2.15x | 2.23x | 2.39x | **2.72x** | **1.14x [1.10, 1.18]** |

Linear suffix decoding with the cache vs without: 1.04x [1.02, 1.06] over all prompts, 0.95x [0.89, 1.00] on chat.
Tokens/forward for tree_new go from 7.68 / 2.19 / 1.83 / 1.32 to 8.09 / 2.80 / 2.41 / 1.41. The cache costs about
0.4 to 1.0 ms more draft time per step, which is why code_edit gains tokens/forward but no time.

The first version of the cache (run 1, `suffix_global`) was slower than no cache: 2.02x vs 2.12x. Two causes, both
found by replaying recorded greedy outputs with `simulate.py --cache`:

1. It kept one suffix tree per cached response. Counts never added up across responses, so every cached continuation
   looked certain (count 1 of 1).
2. Local and cached drafts competed on score, and those over-confident cached drafts replaced correct local ones
   (code_edit: 6.43 → 5.74 tokens/forward on the dev split).

Now all responses go into a shared suffix tree, so a continuation seen in many responses outranks one seen once, and
the sources are combined instead of ranked: linear drafts use the shared tree only when the local tree has no
continuation; tree drafts are the union of both draft trees, verified in one forward. Memory stays bounded without
node-level deletion by keeping the shared tree as two generations and dropping the older one as a whole. The policy
was chosen on the dev split (tokens/forward for tree drafts: 8.11 → 8.60, 1.97 → 2.53, 1.83 → 2.38, 1.34 → 1.44) and
then timed once on the test prompts above.

Limits: the gain depends on responses repeating each other. It is large when requests of one kind follow each other
(HumanEval solutions, GSM8K step-by-step answers) and absent on mixed chat prompts. Batched or concurrent serving was
not measured.

## What moved the needle

1. **Fixing the KV cache bug** (accepted non-first branches kept rejected tokens' KV). Required for correctness; before
   the fix tree outputs diverged from greedy on 4/4 test prompts.
2. **Uncapping draft size.** The suffix-tree draft was capped at `3 * match_len + 5` nodes. A forward of Qwen3-8B on
   H100 costs about the same for 1 or 129 query tokens (see below), so the cap threw away free verification capacity.
   Policy tuned offline on a disjoint dev split with `simulate.py`, which replays greedy outputs exactly and predicted
   the measured tokens/forward within ~3%.
3. **A shared cross-request tree with merged drafts** (previous section).

## What did not

- Larger trees (256 nodes, branch 16): +0.9% (n.s.). Drafting from the suffix tree is saturated — the true
  continuation usually is not in the tree at all.
- Merging drafts from several match lengths: worse (long matches with count 1 are over-confident).
- A lean per-step path (no logits-processor pre-filter, host token mirror, one-shot mask build): correct, but no
  measurable change in ms/step.
- Folding query heads instead of `repeat_kv` for masked GQA attention: 0.997x (n.s.). A first 5-prompt test showed
  +7.7%, which was noise.
- Skipping the shared-tree lookup when the local draft already fills the node budget: saves ~0.1 ms of draft time per
  step (laptop CPU) and loses a little tokens/forward in the dev replay; not adopted.

## Baseline sensitivity: compiled vanilla decoding

All speedups above are against eager `generate`, which is CPU/launch bound: GPU kernels are busy ~50% of wall time,
and the per-token time moved between 20 and 38 ms across containers with different host CPUs.
`phase0_compile.py` measures what a static KV cache plus `torch.compile` change (`results/phase0.json`, second run in
brackets from the log of a container with a slower host):

| | eager | static cache + compile |
|---|---|---|
| vanilla greedy decoding | 19.7 ms/token (30.2) | 9.1 ms/token (10.1) |
| forward, 1 query token on a 1536-token cache | 21.4 ms | 11.1 ms (`reduce-overhead`) |
| forward, 129 query tokens with a tree mask | 21.0 ms | 12.7 ms |

- A compiled baseline is 2.2x to 3.0x faster than the eager one by itself.
- Verification stays almost free under CUDA graphs: 129 query tokens cost 1.14x one token (1.09x in the other run).
  Speculation and compilation therefore stack.
- The speculative paths in this fork are **not** compiled. Compiled vanilla decoding reached 99 to 110 tok/s (one
  336-token prompt, 256 new tokens). The eager speculative methods in v3 are still ahead of that on code_edit
  (238 to 292 tok/s) and behind on the other three tasks (53 to 72 tok/s).
- Projection for a compiled verification forward, from tokens/forward above, the compiled forward times and 1.5 ms
  of host work per step: tree_new 4.9x on code_edit, 1.4x on humaneval, 1.2x on gsm8k, 0.85x on chat, relative to
  compiled vanilla decoding. This is an estimate, not a measurement; it needs static-cache support in the tree
  verifier (in-place KV compaction, fixed mask width, bucketed draft sizes).

## Where the remaining time goes

Speculative steps cost ~3.5 ms more than vanilla steps (e.g. 26.3 vs 22.2 ms on code_edit) for every method (PLD,
suffix, tree alike): ~1.5 ms of generation-loop work and ~2 ms more CPU time inside the model forward for q > 1.

Correctness: in float32 all methods reproduce vanilla greedy exactly, with and without logits processors, EOS,
max-length limits and the cross-request cache (`tests/generation/test_candidate_generator.py`). On Qwen3-0.6B in
float32 on CPU (`lossless_check.py`), suffix and tree decoding reproduced greedy output token for token on 15 of 15
CanItEdit prompts, and again on 13 of 13 with the cross-request cache. In bf16 ~50% of outputs are bit-identical for every speculative method
including upstream PLD: query length changes bf16 reduction order and greedy decoding amplifies near-ties.
