"""Phase 0 (go/no-go) for CUDA graphs / torch.compile on the verification forward.

Questions:
  1. Once launch/CPU overhead is gone, does a forward with q=129 query tokens still cost about the same as q=1?
  2. How much faster does vanilla greedy decoding get with a static cache + compile?

    modal run bench_tree_spec/phase0_compile.py
    modal run bench_tree_spec/phase0_compile.py --modes eager_dynamic,compile_ro --ctxs 1536

Self-contained on purpose (no import of modal_app.py, so nothing has to be mounted besides src/transformers).
Inductor's cache lives on the HF volume, so a rerun skips most of the compilation time.
"""

import json
import os
import statistics
import time
from pathlib import Path

import modal


REPO_ROOT = Path(__file__).resolve().parent.parent
GPU = os.environ.get("BENCH_GPU", "H100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "huggingface-hub>=1.5.0,<2.0",
        "tokenizers>=0.22.0,<=0.23.0",
        "safetensors>=0.4.3",
        "regex>=2025.10.22",
        "numpy",
        "packaging",
        "pyyaml",
        "tqdm",
        "typer",
        "accelerate",
    )
    .env(
        {
            "PYTHONPATH": "/opt/tf",
            "HF_HUB_CACHE": "/hf",
            "TORCHINDUCTOR_CACHE_DIR": "/hf/inductor-cache",
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        }
    )
    .add_local_dir(REPO_ROOT / "src" / "transformers", "/opt/tf/transformers", ignore=["**/__pycache__"])
)
hf_cache = modal.Volume.from_name("llm-bench-hf-cache")
app = modal.App("tree-spec-phase0", image=image)

# draft nodes 0/16/32/64/128 plus the root token: the bucket sizes phase 1 would capture
QUERY_LENS = (1, 17, 33, 65, 129)
MAX_CACHE_LEN = 2048  # covers ctx 1536 + 129; the mask is always this wide, as for a real static cache
ALL_MODES = ("eager_dynamic", "eager_static", "compile_default", "compile_ro")

# tokens per forward and eager ms/step from results/qwen3-8b-h100-v3.json (pooled over prompts), for the projection
V3 = {
    "code_edit": {"suffix": 6.21, "tree_new": 7.68},
    "humaneval": {"suffix": 2.05, "tree_new": 2.19},
    "gsm8k": {"suffix": 1.63, "tree_new": 1.83},
    "chat": {"suffix": 1.26, "tree_new": 1.32},
}


def _tree_like_mask(ctx, q, kv_width, dtype, device):
    """Additive 4D mask of shape (1, 1, q, kv_width): the cached prefix plus a causal block over the new tokens.

    A real tree mask has the same shape and sparsity class; SDPA cost does not depend on which entries are masked.
    """
    import torch

    allowed = torch.zeros((q, kv_width), dtype=torch.bool, device=device)
    allowed[:, :ctx] = True
    allowed[:, ctx : ctx + q] = torch.ones((q, q), dtype=torch.bool, device=device).tril()
    mask = torch.zeros((1, 1, q, kv_width), dtype=dtype, device=device)
    return mask.masked_fill_(~allowed, torch.finfo(dtype).min)


def _median_ms(fn, warmup, iters):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        # wall clock with a sync per call: the decoding loop syncs every step (`.tolist()` on the accepted tokens)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


@app.function(gpu=GPU, volumes={"/hf": hf_cache}, timeout=2 * 3600)
def phase0(model_id: str, ctxs: list[int], modes: list[str], gen_tokens: int, iters: int):
    import torch

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache

    assert transformers.__file__.startswith("/opt/tf"), transformers.__file__
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 64

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to("cuda").eval()
    dtype, device = model.dtype, model.device
    report = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "forward_ms": {}, "compile_s": {}}

    # one static cache for every static/compiled measurement: a new cache object means new static addresses, i.e.
    # dynamo guard failures and fresh cudagraph recordings
    static_cache = StaticCache(config=model.config, max_cache_len=MAX_CACHE_LEN)
    compiled = {
        "eager_static": model.__call__,
        "compile_default": torch.compile(model.__call__, mode="default", fullgraph=False, dynamic=False),
        # what `generate` uses for static caches (CompileConfig defaults: reduce-overhead, i.e. cudagraph trees)
        "compile_ro": torch.compile(model.__call__, mode="reduce-overhead", fullgraph=False, dynamic=False),
    }

    # ---- 1. forward latency vs query length on top of a ctx-long cache -------------------------------------------
    for mode in modes:
        for ctx in ctxs:
            prefix = torch.randint(0, 1000, (1, ctx), device=device)
            with torch.no_grad():
                if mode == "eager_dynamic":
                    cache = DynamicCache(config=model.config)
                    model(prefix, past_key_values=cache, use_cache=True)
                else:
                    static_cache.reset()
                    model(prefix, past_key_values=static_cache, use_cache=True)  # eager prefill, as in generate
                    cache = static_cache

                for q in QUERY_LENS:
                    width = ctx + q if mode == "eager_dynamic" else MAX_CACHE_LEN
                    inputs = {
                        "input_ids": torch.randint(0, 1000, (1, q), device=device),
                        "attention_mask": _tree_like_mask(ctx, q, width, dtype, device),
                        "position_ids": torch.arange(ctx, ctx + q, device=device).unsqueeze(0),
                        "past_key_values": cache,
                        "use_cache": True,
                        "logits_to_keep": q,
                    }

                    if mode == "eager_dynamic":

                        def step():
                            model(**inputs)
                            cache.crop(ctx)

                    else:
                        forward = compiled[mode]

                        def step():
                            if mode == "compile_ro":
                                torch.compiler.cudagraph_mark_step_begin()
                            out = forward(**inputs)
                            out.logits.float()  # consume the output before the next replay overwrites it
                            for layer in cache.layers:  # in place: the static address must not change
                                layer.cumulative_length.fill_(ctx)

                    t0 = time.perf_counter()
                    step()  # first call compiles this shape
                    torch.cuda.synchronize()
                    report["compile_s"][f"{mode}_ctx{ctx}_q{q}"] = time.perf_counter() - t0
                    ms = _median_ms(step, warmup=5, iters=iters)
                    report["forward_ms"][f"{mode}_ctx{ctx}_q{q}"] = ms
                    print(f"{mode:<16} ctx={ctx:<5} q={q:<4} {ms:7.2f} ms", flush=True)

    # ---- 2. KV compaction as done after a tree step (one gather + copy per layer and K/V), eager -------------------
    ctx = ctxs[-1]
    accepted = [0, 3, 9, 17, 40, 63, 90, 127]  # a non-first branch of depth 8
    with torch.no_grad():
        static_cache.reset()
        model(torch.randint(0, 1000, (1, ctx + 129), device=device), past_key_values=static_cache, use_cache=True)

        def compact():
            keep = torch.tensor(accepted, device=device) + ctx
            for layer in static_cache.layers:
                layer.keys[..., ctx : ctx + len(accepted), :] = layer.keys[..., keep, :]
                layer.values[..., ctx : ctx + len(accepted), :] = layer.values[..., keep, :]
                layer.cumulative_length.fill_(ctx + len(accepted))

        report["compact_ms"] = _median_ms(compact, warmup=3, iters=iters)
        print(f"compaction (8 nodes, eager): {report['compact_ms']:.2f} ms", flush=True)

    # ---- 3. vanilla greedy decoding, eager DynamicCache vs static cache + auto compile ----------------------------
    messages = [
        {
            "role": "user",
            "content": "Rewrite this function to be iterative and explain each step:\n"
            + "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n" * 12,
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    report["gen_prompt_tokens"] = inputs.input_ids.shape[1]
    # fixed output length so both variants decode exactly `gen_tokens` tokens
    gen = {
        "max_new_tokens": gen_tokens,
        "min_new_tokens": gen_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }
    report["vanilla_ms_per_token"] = {}
    for name, extra in [("eager_dynamic", {}), ("static_compiled", {"cache_implementation": "static"})]:
        t0 = time.perf_counter()
        model.generate(**inputs, **gen, **extra)  # warmup / compilation
        report["compile_s"][f"generate_{name}"] = time.perf_counter() - t0
        lat = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.generate(**inputs, **gen, **extra)
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
        report["vanilla_ms_per_token"][name] = statistics.median(lat) / gen_tokens * 1e3
        print(f"vanilla {name:<16} {report['vanilla_ms_per_token'][name]:.2f} ms/token", flush=True)

    hf_cache.commit()  # keep the inductor cache for the next run
    return report


def _verdict(report, ctx, host_ms):
    fwd = report["forward_ms"]
    mode = "compile_ro" if f"compile_ro_ctx{ctx}_q1" in fwd else "eager_static"
    t = {q: fwd[f"{mode}_ctx{ctx}_q{q}"] for q in QUERY_LENS}
    ratio = t[129] / t[1]
    van = report["vanilla_ms_per_token"]
    print(
        f"\n== {mode}, ctx={ctx}: q129/q1 = {ratio:.2f}  (q17 {t[17] / t[1]:.2f}, q33 {t[33] / t[1]:.2f}, "
        f"q65 {t[65] / t[1]:.2f})"
    )
    print(
        f"== vanilla: eager {van['eager_dynamic']:.2f} -> static+compile {van['static_compiled']:.2f} ms/token "
        f"({van['eager_dynamic'] / van['static_compiled']:.2f}x)"
    )

    # projected speedup over *compiled* vanilla, assuming tokens/forward from v3 and `host_ms` of non-graphed work per
    # speculative step (drafting, acceptance, mask upload, compaction); suffix drafts <= 32 nodes -> q bucket 33
    t_van = van["static_compiled"]
    print(f"\nprojection vs compiled vanilla (host overhead {host_ms} ms/step):")
    for task, tpf in V3.items():
        suffix = tpf["suffix"] * t_van / (t[33] + host_ms)
        tree = tpf["tree_new"] * t_van / (t[129] + host_ms)
        print(f"  {task:<10} suffix {suffix:5.2f}x  tree {tree:5.2f}x  tree/suffix {tree / suffix:4.2f}")
    verdict = (
        "GO" if ratio <= 1.3 else "GO, but retune buckets/policy with a cost(q) model" if ratio <= 1.7 else "STOP"
    )
    print(f"\nverdict on q129/q1: {verdict}")


@app.local_entrypoint()
def main(
    model_id: str = "Qwen/Qwen3-8B",
    ctxs: str = "512,1536",
    modes: str = ",".join(ALL_MODES),
    gen_tokens: int = 256,
    iters: int = 20,
    host_ms: float = 1.5,
    out: str = "bench_tree_spec/results/phase0.json",
):
    ctx_list = [int(c) for c in ctxs.split(",")]
    report = phase0.remote(model_id, ctx_list, modes.split(","), gen_tokens, iters)
    Path(out).write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")
    _verdict(report, ctx_list[-1], host_ms)
