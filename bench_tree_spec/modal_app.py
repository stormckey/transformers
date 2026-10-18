"""Tree-spec / suffix decoding benchmark on Modal.

Every prompt is run through every method back to back inside one GPU container (method order rotated per prompt),
so each method sees the same prompts, GPU and thermals. Speedups are computed per prompt against vanilla greedy
decoding in the same container.

    modal run bench_tree_spec/modal_app.py --tasks code_edit,humaneval,gsm8k,chat --n 20
"""

import json
import os
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
        "datasets>=4.0",
        "accelerate",
    )
    .env({"PYTHONPATH": "/opt/tf", "HF_HUB_CACHE": "/hf", "HF_DATASETS_CACHE": "/hf/datasets"})
    # local transformers checkout shadows the pip one (pip is only there to pull its dependencies)
    .add_local_dir(REPO_ROOT / "src" / "transformers", "/opt/tf/transformers", ignore=["**/__pycache__"])
)
hf_cache = modal.Volume.from_name("llm-bench-hf-cache")
app = modal.App("tree-spec-bench", image=image)


def _methods():
    from transformers.generation.suffix_tree import SuffixDecodingCache

    methods = {
        "vanilla": lambda: {},
        "pld": lambda: {"prompt_lookup_num_tokens": 10},
        "suffix": lambda: {"suffix_decoding_num_tokens": 32},
        "tree_b2_n16": lambda: {"tree_spec_decoding_num_tokens": 16, "tree_spec_decoding_branch_factor": 2},
        "tree_b2_n32": lambda: {"tree_spec_decoding_num_tokens": 32, "tree_spec_decoding_branch_factor": 2},
        "tree_b4_n32": lambda: {"tree_spec_decoding_num_tokens": 32, "tree_spec_decoding_branch_factor": 4},
        "tree_b4_n64": lambda: {"tree_spec_decoding_num_tokens": 64, "tree_spec_decoding_branch_factor": 4},
        # policy tuned offline on the dev split with bench_tree_spec/simulate.py
        "tree_new": lambda: {
            "tree_spec_decoding_num_tokens": 128,
            "tree_spec_decoding_branch_factor": 8,
            "tree_spec_decoding_min_prob": 0.02,
            "tree_spec_decoding_max_spec_factor": 1e9,
            "tree_spec_decoding_max_spec_offset": 1e9,
        },
        "tree_big": lambda: {
            "tree_spec_decoding_num_tokens": 256,
            "tree_spec_decoding_branch_factor": 16,
            "tree_spec_decoding_min_prob": 0.01,
            "tree_spec_decoding_max_spec_factor": 1e9,
            "tree_spec_decoding_max_spec_offset": 1e9,
        },
        "tree_b4_n64_p02": lambda: {
            "tree_spec_decoding_num_tokens": 64,
            "tree_spec_decoding_branch_factor": 4,
            "tree_spec_decoding_min_prob": 0.02,
        },
    }
    # `<method>_fold`: same kwargs, but masked GQA attention folds query heads instead of repeating K/V
    for name in list(methods):
        methods[f"{name}_fold"] = methods[name]
    # cross-request variants: one persistent cache per method, shared across the prompts of a task
    caches = {
        "suffix_global": SuffixDecodingCache(max_depth=64, max_cached_requests=64),
        "tree_global_b4_n32": SuffixDecodingCache(max_depth=64, max_cached_requests=64),
    }
    methods["suffix_global"] = lambda: {
        "suffix_decoding_num_tokens": 32,
        "suffix_decoding_cache": caches["suffix_global"],
    }
    methods["tree_global_b4_n32"] = lambda: {
        "tree_spec_decoding_num_tokens": 32,
        "tree_spec_decoding_branch_factor": 4,
        "tree_spec_decoding_cache": caches["tree_global_b4_n32"],
    }
    return methods


def _load_prompts(task: str, n: int, tokenizer, seed: int = 0, offset: int = 0) -> list[str]:
    """Returns prompts `offset .. offset + n` of the shuffled task (test set = offset 0, dev set = offset 30)."""
    from datasets import load_dataset

    if task == "code_edit":
        ds = load_dataset("nuprl/CanItEdit", split="test").shuffle(seed=seed)
        msgs = [
            f"Edit the following Python code according to the instruction. Output the full edited code only.\n\n"
            f"Instruction: {row['instruction_descriptive']}\n\n```python\n{row['before']}\n```"
            for row in ds
        ]
    elif task == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=seed)
        msgs = [f"Complete the following Python function.\n\n```python\n{row['prompt']}```" for row in ds]
    elif task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed)
        msgs = [f"Solve step by step.\n\n{row['question']}" for row in ds]
    elif task == "chat":
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train").shuffle(seed=seed)
        msgs = [row["prompt"][0] for row in ds]
    else:
        raise ValueError(task)

    prompts = []
    for msg in msgs:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        # keep prompts reasonably short so one container finishes in bounded time
        if len(tokenizer(text).input_ids) <= 2048:
            prompts.append(text)
        if len(prompts) == offset + n:
            break
    return prompts[offset:]


class _Probe:
    """Counts target forward passes, candidate-generation time and accepted lengths during one generate call."""

    def __init__(self, model):
        from transformers.generation import candidate_generator as cg

        self.reset()
        model.register_forward_pre_hook(self._on_forward)

        probe = self
        for cls in (
            cg.PromptLookupCandidateGenerator,
            cg.SuffixDecodingCandidateGenerator,
            cg.TreeSpecDecodingCandidateGenerator,
        ):
            orig = cls.get_candidates

            def timed(gen, input_ids, _orig=orig):
                t0 = time.perf_counter()
                out = _orig(gen, input_ids)
                probe.draft_time += time.perf_counter() - t0
                n_draft = out[0].shape[1] - input_ids.shape[1]
                probe.proposed += n_draft
                draft = getattr(gen, "tree_draft", None)
                probe.tree_steps += int(draft is not None and draft.is_tree)
                return out

            cls.get_candidates = timed

        for cls in (cg.PromptLookupCandidateGenerator, cg.SuffixDecodingCandidateGenerator):
            orig_update = cls.update_candidate_strategy

            def counted(gen, input_ids, scores, num_matches, _orig=orig_update):
                probe.accepted += int(num_matches)
                return _orig(gen, input_ids, scores, num_matches)

            cls.update_candidate_strategy = counted

    def reset(self):
        self.forwards = 0
        self.draft_time = 0.0
        self.proposed = 0
        self.accepted = 0
        self.tree_steps = 0

    def _on_forward(self, module, args):
        self.forwards += 1


@app.function(gpu=GPU, volumes={"/hf": hf_cache}, timeout=6 * 3600)
def run_task(task: str, n: int, model_id: str, max_new_tokens: int, method_names: list[str] | None = None):
    import torch

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert transformers.__file__.startswith("/opt/tf"), transformers.__file__
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to("cuda").eval()
    probe = _Probe(model)

    methods = _methods()
    if method_names:
        methods = {k: v for k, v in methods.items() if k in method_names or k == "vanilla"}
    names = list(methods)

    prompts = _load_prompts(task, n, tokenizer)
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False, "pad_token_id": tokenizer.eos_token_id}

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    sdpa_default, sdpa_fold = ALL_ATTENTION_FUNCTIONS["sdpa"], _sdpa_variant("fold_heads")

    def run_one(prompt, name):
        ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_fold if name.endswith("_fold") else sdpa_default
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        probe.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs, **methods[name]())
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        # prompt lookup may overshoot max_new_tokens by a few tokens: compare and count on the same budget
        new_tokens = out[0, inputs.input_ids.shape[1] :].tolist()[:max_new_tokens]
        return {
            "latency": latency,
            "new_tokens": len(new_tokens),
            "tokens": new_tokens,
            "forwards": probe.forwards,
            "draft_time": probe.draft_time,
            "proposed": probe.proposed,
            "accepted": probe.accepted,
            "tree_steps": probe.tree_steps,
        }

    # warmup: CUDA context, kernels autotuning, allocator
    warm_prompt = _load_prompts("chat", 1, tokenizer, seed=123)[0]
    for name in names:
        if "global" not in name:
            run_one(warm_prompt, name)

    records = []
    for i, prompt in enumerate(prompts):
        order = names[i % len(names) :] + names[: i % len(names)]
        results = {name: run_one(prompt, name) for name in order}
        ref = results["vanilla"]["tokens"]
        for name, r in results.items():
            toks = r.pop("tokens")
            same_prefix = 0
            for a, b in zip(toks, ref):
                if a != b:
                    break
                same_prefix += 1
            r.update(
                task=task,
                prompt_idx=i,
                method=name,
                exact_match=toks == ref,
                match_prefix=same_prefix,
                prompt_tokens=len(tokenizer(prompt).input_ids),
            )
            records.append(r)
        v = results["vanilla"]["latency"]
        print(
            f"[{task} {i + 1}/{len(prompts)}] "
            + " ".join(f"{k}={v / r['latency']:.2f}x" for k, r in results.items() if k != "vanilla"),
            flush=True,
        )

    return {
        "task": task,
        "model": model_id,
        "gpu": torch.cuda.get_device_name(),
        "max_new_tokens": max_new_tokens,
        "records": records,
    }


@app.function(gpu=GPU, volumes={"/hf": hf_cache}, timeout=3600)
def dump_vanilla(task: str, n: int, offset: int, model_id: str, max_new_tokens: int):
    """Greedy vanilla outputs, used to replay drafting policies offline (greedy output does not depend on drafts)."""
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to("cuda").eval()
    rows = []
    for prompt in _load_prompts(task, n, tokenizer, offset=offset):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        rows.append({"prompt": inputs.input_ids[0].tolist(), "output": out[0, inputs.input_ids.shape[1] :].tolist()})
    return {"task": task, "rows": rows}


@app.function(gpu=GPU, volumes={"/hf": hf_cache}, timeout=3600)
def profile(model_id: str):
    """Forward latency vs query length, and a torch.profiler trace of the suffix decoding loop."""
    import torch
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as tprofile

    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to("cuda").eval()
    report = {}

    # 1. pure forward time vs query length on top of a cache
    fwd = {}
    for ctx in (512, 1536):
        prefix = torch.randint(0, 1000, (1, ctx), device="cuda")
        for q in (1, 2, 4, 8, 16, 33, 65, 129):
            with torch.inference_mode():
                cache = DynamicCache(config=model.config)
                model(prefix, past_key_values=cache, use_cache=True)
                x = torch.randint(0, 1000, (1, q), device="cuda")
                times = []
                for _ in range(12):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    model(x, past_key_values=cache, use_cache=True, logits_to_keep=q)
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end))
                    cache.crop(ctx)
            fwd[f"ctx{ctx}_q{q}"] = sorted(times[2:])[len(times[2:]) // 2]
    report["forward_ms"] = fwd
    print(json.dumps(fwd, indent=1), flush=True)

    # 2. where does the time of a generate() call go
    prompt = _load_prompts("code_edit", 1, tokenizer, offset=40)[0]
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    for name, kw in [("vanilla", {}), ("suffix", {"suffix_decoding_num_tokens": 32})]:
        gen = dict(max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id, **kw)
        with torch.inference_mode():
            model.generate(**inputs, **gen)
            torch.cuda.synchronize()
            with tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                t0 = time.perf_counter()
                model.generate(**inputs, **gen)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
        events = prof.key_averages()
        gpu_busy = sum(e.self_device_time_total for e in events) / 1e6
        table = events.table(sort_by="cpu_time_total", row_limit=40, max_name_column_width=70)
        report[name] = {"wall_s": wall, "gpu_kernel_s": gpu_busy, "table": table}
        print(f"== {name}: wall {wall:.2f}s, gpu kernels {gpu_busy:.2f}s", flush=True)
        print(table, flush=True)
    return report


@app.local_entrypoint()
def main(
    tasks: str = "code_edit,humaneval,gsm8k,chat",
    n: int = 20,
    model: str = "Qwen/Qwen3-8B",
    max_new_tokens: int = 384,
    methods: str = "",
    out: str = "",
):
    method_names = [m for m in methods.split(",") if m] or None
    task_list = tasks.split(",")
    results = list(
        run_task.starmap([(t, n, model, max_new_tokens, method_names) for t in task_list], return_exceptions=True)
    )
    out_path = Path(out or f"bench_tree_spec/results/{time.strftime('%Y%m%d-%H%M%S')}-{model.split('/')[-1]}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = [r for r in results if isinstance(r, dict)]
    for r in results:
        if not isinstance(r, dict):
            print("TASK FAILED:", repr(r))
    out_path.write_text(json.dumps(ok))
    print("wrote", out_path)


@app.function(gpu=GPU, volumes={"/hf": hf_cache}, timeout=1800)
def cprofile_steps(model_id: str = "Qwen/Qwen3-8B", max_new_tokens: int = 256):
    """Host-side time breakdown (cProfile) of one generate call per method; the loop is CPU/launch bound."""
    import cProfile
    import io
    import pstats

    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to("cuda").eval()
    prompt = _load_prompts("chat", 1, tokenizer, offset=45)[0]
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    methods = _methods()
    report = {}
    for name in ("vanilla", "suffix", "tree_new"):
        gen = {"max_new_tokens": max_new_tokens, "do_sample": False, "pad_token_id": tokenizer.eos_token_id}
        gen.update(methods[name]())
        with torch.inference_mode():
            model.generate(**inputs, **gen)
            torch.cuda.synchronize()
            prof = cProfile.Profile()
            prof.enable()
            out = model.generate(**inputs, **gen)
            torch.cuda.synchronize()
            prof.disable()
        buf = io.StringIO()
        stats = pstats.Stats(prof, stream=buf)
        stats.sort_stats("tottime").print_stats(35)
        loop = "_assisted_decoding" if name != "vanilla" else "_sample"
        stats.sort_stats("cumulative").print_callees(loop)
        report[name] = {"new_tokens": out.shape[1] - inputs.input_ids.shape[1], "text": buf.getvalue()}
        print(f"=== {name}", report[name]["new_tokens"], flush=True)
        print(buf.getvalue(), flush=True)
    return report


@app.local_entrypoint()
def cprof():
    report = cprofile_steps.remote()
    Path("bench_tree_spec/results/cprofile.json").write_text(json.dumps(report))


def _sdpa_variant(kind: str):
    """Alternative SDPA wrappers for masked GQA attention (tree verification always passes a 4D mask)."""
    import torch

    from transformers.integrations import sdpa_attention as sa

    def forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
        if attention_mask is None or not hasattr(module, "num_key_value_groups"):
            return sa.sdpa_attention_forward(module, query, key, value, attention_mask, dropout, scaling, is_causal)
        n_rep = module.num_key_value_groups
        if kind == "gqa_flag":
            out = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, scale=scaling, enable_gqa=True
            )
        elif kind == "fold_heads":
            b, hq, q, d = query.shape
            hkv = key.shape[1]
            folded = query.reshape(b, hkv, n_rep * q, d)
            mask = attention_mask.expand(b, 1, q, -1).repeat(1, 1, n_rep, 1)
            out = torch.nn.functional.scaled_dot_product_attention(folded, key, value, attn_mask=mask, scale=scaling)
            out = out.reshape(b, hq, q, d)
        else:
            raise ValueError(kind)
        return out.transpose(1, 2).contiguous(), None

    return forward


@app.function(gpu=GPU, volumes={"/hf": hf_cache}, timeout=1800)
def attn_variants(model_id: str = "Qwen/Qwen3-8B"):
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to("cuda").eval()
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    variants = {
        "repeat_kv": original,
        "gqa_flag": _sdpa_variant("gqa_flag"),
        "fold_heads": _sdpa_variant("fold_heads"),
    }
    report = {}

    # 1. wall time per forward, q draft tokens on top of a 1200-token cache, tree-style 4D mask
    ctx = 1200
    prefix = torch.randint(0, 1000, (1, ctx), device="cuda")
    for q in (1, 17, 65, 129):
        for name, fn in list(variants.items()) + [("no_mask", original)]:
            ALL_ATTENTION_FUNCTIONS["sdpa"] = fn
            with torch.inference_mode():
                cache = DynamicCache(config=model.config)
                model(prefix, past_key_values=cache, use_cache=True)
                x = torch.randint(0, 1000, (1, q), device="cuda")
                mask = torch.zeros((1, 1, q, ctx + q), dtype=torch.bfloat16, device="cuda")
                mask[..., ctx:] = torch.full((q, q), torch.finfo(torch.bfloat16).min, device="cuda").triu(1)
                kw = {} if name == "no_mask" else {"attention_mask": mask}
                times = []
                for _ in range(15):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    model(x, past_key_values=cache, use_cache=True, logits_to_keep=q, **kw)
                    torch.cuda.synchronize()
                    times.append(1000 * (time.perf_counter() - t0))
                    cache.crop(ctx)
                # check the variant is exact against the reference repeat_kv path
            report[f"q{q}_{name}"] = sorted(times[3:])[len(times[3:]) // 2]
            print(f"q={q:<4} {name:<11} {report[f'q{q}_{name}']:.2f} ms", flush=True)
    ALL_ATTENTION_FUNCTIONS["sdpa"] = original

    # 2. numerical check of the variants against repeat_kv on a tree mask
    with torch.inference_mode():
        x = torch.randint(0, 1000, (1, 65), device="cuda")
        mask = torch.zeros((1, 1, 65, ctx + 65), dtype=torch.bfloat16, device="cuda")
        mask[..., ctx:] = torch.full((65, 65), torch.finfo(torch.bfloat16).min, device="cuda").triu(1)
        outs = {}
        for name, fn in variants.items():
            ALL_ATTENTION_FUNCTIONS["sdpa"] = fn
            cache = DynamicCache(config=model.config)
            model(prefix, past_key_values=cache, use_cache=True)
            outs[name] = model(x, attention_mask=mask, past_key_values=cache).logits.float()
        for name in variants:
            diff = (outs[name] - outs["repeat_kv"]).abs().max().item()
            agree = (outs[name].argmax(-1) == outs["repeat_kv"].argmax(-1)).float().mean().item()
            report[f"diff_{name}"] = diff
            print(f"{name}: max|dlogit| {diff:.4f}  argmax agree {agree:.3f}", flush=True)
    ALL_ATTENTION_FUNCTIONS["sdpa"] = original

    # 3. end to end, tree_new on 5 chat prompts
    methods = _methods()
    prompts = _load_prompts("chat", 5, tokenizer, offset=50)
    probe = _Probe(model)
    for name in ("repeat_kv", "gqa_flag", "fold_heads"):
        ALL_ATTENTION_FUNCTIONS["sdpa"] = variants[name]
        lat = fwd = 0
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            gen = dict(
                max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id, **methods["tree_new"]()
            )
            with torch.inference_mode():
                if i == 0:
                    model.generate(**inputs, **gen)
                probe.reset()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model.generate(**inputs, **gen)
                torch.cuda.synchronize()
                lat += time.perf_counter() - t0
                fwd += probe.forwards
        report[f"e2e_{name}_ms_per_fwd"] = 1000 * lat / fwd
        print(f"e2e tree_new with {name}: {1000 * lat / fwd:.2f} ms/forward", flush=True)
    ALL_ATTENTION_FUNCTIONS["sdpa"] = original
    return report


@app.local_entrypoint()
def attn():
    Path("bench_tree_spec/results/attn_variants.json").write_text(json.dumps(attn_variants.remote()))
