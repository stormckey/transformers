# Tree-spec vs suffix decoding — Qwen3-8B, H100

Raw data: `results/qwen3-8b-h100.json` (run with `modal run bench_tree_spec/modal_app.py --n 30 --max-new-tokens 384`).

Setup: bf16, SDPA, greedy, batch size 1, 30 prompts per task, every prompt runs all methods back to back in one
container (method order rotated per prompt). Speedup = geometric mean over prompts of vanilla latency / method latency.

| task | vanilla tok/s | PLD | suffix (linear) | tree b4 n64 | tree vs suffix (paired, 95% CI) |
|---|---|---|---|---|---|
| code_edit (CanItEdit) | 43.6 | 3.90x | 6.60x | **7.28x** | 1.10x [1.07, 1.14] |
| humaneval | 36.3 | 1.73x | 1.89x | **1.97x** | 1.04x [1.00, 1.10] |
| gsm8k | 54.8 | 1.41x | 1.43x | **1.52x** | 1.06x [1.05, 1.08] |
| chat (MT-Bench) | 40.3 | 1.11x | 1.14x | **1.18x** | 1.04x [1.01, 1.07] |
| all | 42.4 | 1.80x | 2.12x | **2.25x** | 1.06x [1.04, 1.08] |

Findings

- Tree verification costs nothing extra per step: ms per target forward is the same for linear suffix and tree drafts
  up to 64 nodes (e.g. code_edit 27.3 vs 27.2 ms). An 8B model at batch size 1 is memory-bound, so the extra draft
  tokens are free.
- The gain is limited by the drafts: only ~30% of steps actually produce a branching draft, and tokens per forward
  rises by 2-8% over linear suffix. Speedup follows tokens per forward almost one to one.
- Wider/bigger trees help (b4 n64 > b4 n32 > b2 n32); b2 n16 is worse than linear suffix n32 on code_edit because it
  caps the draft length. Lowering `min_prob` to 0.02 changes nothing — the `match_len * 3 + 5` length cap binds first.
- The cross-request cache (`*_global`) is slower here: it lowers acceptance on these heterogeneous prompts and adds
  ~0.4 ms of Python draft time per step.
- Draft time (pure-Python suffix tree) is ~0.8 ms per step, ~3% of a forward.

Correctness: in float32 every method reproduces vanilla greedy exactly (see tests in
`tests/generation/test_candidate_generator.py`). In bf16 only ~50% of outputs are bit-identical to vanilla for *all*
speculative methods including upstream PLD — changing the query length changes bf16 reduction order, and greedy
decoding amplifies near-ties.
