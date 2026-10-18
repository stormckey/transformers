# Tree-spec vs suffix decoding — Qwen3-8B, H100

Setup: bf16, SDPA, greedy, batch size 1, 30 test prompts per task (same prompts in every run), max 384 new tokens.
Every prompt runs all methods back to back in one container with the method order rotated per prompt; speedup is the
geometric mean over prompts of vanilla latency / method latency, and method-vs-method numbers are paired per prompt
(95% bootstrap CI).

## Final numbers (`results/qwen3-8b-h100-v3.json`)

| task | PLD | suffix (linear) | tree, old policy (run 1) | **tree_new** | tree_new vs suffix |
|---|---|---|---|---|---|
| code_edit (CanItEdit) | 3.90x | 6.48x | 7.28x | **8.76x** | 1.36x |
| humaneval | 1.72x | 1.87x | 1.97x | **1.94x** | 1.05x |
| gsm8k | 1.44x | 1.45x | 1.52x | **1.58x** | 1.10x |
| chat (MT-Bench) | 1.13x | 1.16x | 1.18x | **1.14x** | 0.98x |
| all | 1.79x | 2.12x | 2.25x | **2.36x** | **1.11x [1.08, 1.15]** |

`tree_new` = `tree_spec_decoding_num_tokens=128, branch_factor=8, min_prob=0.02, max_spec_factor=max_spec_offset=1e9`.
The tree_new/suffix ratio reproduced exactly across two independent runs (v2: 1.112x, v3: 1.113x).

## What moved the needle

1. **Fixing the KV cache bug** (accepted non-first branches kept rejected tokens' KV). Required for correctness; before
   the fix tree outputs diverged from greedy on 4/4 test prompts.
2. **Uncapping draft size.** The suffix-tree draft was capped at `3 * match_len + 5` nodes. A forward of Qwen3-8B on
   H100 costs the same ~22 ms for 1 or 129 query tokens (launch/CPU bound), so the cap threw away free verification
   capacity. Policy tuned offline on a disjoint dev split with `simulate.py`, which replays greedy outputs exactly and
   predicted the measured tokens/forward within ~3%.

## What did not

- Larger trees (256 nodes, branch 16): +0.9% (n.s.). Drafting from the suffix tree is saturated — the true
  continuation usually is not in the tree at all.
- Merging drafts from several match lengths: worse (long matches with count 1 are over-confident).
- A lean per-step path (no logits-processor pre-filter, host token mirror, one-shot mask build): correct, but no
  measurable change in ms/step.
- Folding query heads instead of `repeat_kv` for masked GQA attention: 0.997x (n.s.). A first 5-prompt test showed
  +7.7%, which was noise.
- Cross-request (global) suffix cache: slower on these heterogeneous prompts (run 1).

## Where the remaining time goes

Speculative steps cost ~3.5 ms more than vanilla steps (e.g. 26.3 vs 22.2 ms on code_edit) for every method (PLD,
suffix, tree alike): ~1.5 ms of generation-loop work and ~2 ms more CPU time inside the model forward for q > 1. The
whole loop is CPU/launch bound (GPU kernels are busy ~50% of wall time in vanilla decoding), so the next big lever is
CUDA graphs / `torch.compile` for the verification forward — which would also speed up the vanilla baseline and has to
be reported as such.

Correctness: in float32 all methods reproduce vanilla greedy exactly, with and without logits processors, EOS and
max-length limits (`tests/generation/test_candidate_generator.py`). In bf16 ~50% of outputs are bit-identical for every
speculative method including upstream PLD: query length changes bf16 reduction order and greedy decoding amplifies
near-ties.
