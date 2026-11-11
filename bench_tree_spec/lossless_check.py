"""Float32 check that suffix and tree decoding reproduce greedy output, on a small real model on CPU.

Uses the CanItEdit prompts of the benchmark (same construction and shuffle seed as modal_app.py) with Qwen3-0.6B, so
it runs on a laptop. `--cache` routes both methods through a cross-request `SuffixDecodingCache`.

    PYTHONPATH=src python bench_tree_spec/lossless_check.py --n 14 --max-new-tokens 256
    PYTHONPATH=src python bench_tree_spec/lossless_check.py --n 12 --max-new-tokens 200 --cache
"""

import argparse
import json

import torch
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import SuffixDecodingCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--n", type=int, default=14)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa").eval()

    prompts = []
    for row in load_dataset("nuprl/CanItEdit", split="test").shuffle(seed=0):
        msg = (
            "Edit the following Python code according to the instruction. Output the full edited code only.\n\n"
            f"Instruction: {row['instruction_descriptive']}\n\n```python\n{row['before']}\n```"
        )
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if len(tokenizer(text).input_ids) <= 2048:
            prompts.append(text)
        if len(prompts) == 30:
            break
    # the first n benchmark prompts, plus the two where tree decoding diverged on Qwen3-8B in bf16
    indices = sorted(set(range(args.n)) | {8, 24})

    methods = {
        "suffix": {"suffix_decoding_num_tokens": 32},
        "tree_new": {
            "tree_spec_decoding_num_tokens": 128,
            "tree_spec_decoding_branch_factor": 8,
            "tree_spec_decoding_min_prob": 0.02,
            "tree_spec_decoding_max_spec_factor": 1e9,
            "tree_spec_decoding_max_spec_offset": 1e9,
        },
    }
    if args.cache:
        methods["suffix"]["suffix_decoding_cache"] = SuffixDecodingCache(max_depth=64, max_cached_requests=64)
        methods["tree_new"]["tree_spec_decoding_cache"] = SuffixDecodingCache(max_depth=64, max_cached_requests=64)

    forwards = [0]
    model.register_forward_pre_hook(lambda module, inputs: forwards.__setitem__(0, forwards[0] + 1))
    gen = {"max_new_tokens": args.max_new_tokens, "do_sample": False, "pad_token_id": tokenizer.eos_token_id}
    exact = dict.fromkeys(methods, 0)
    for i in indices:
        inputs = tokenizer(prompts[i], return_tensors="pt")
        n_in = inputs.input_ids.shape[1]
        with torch.inference_mode():
            reference = model.generate(**inputs, **gen)[0, n_in:].tolist()[: args.max_new_tokens]
            row = {"prompt_idx": i, "prompt_tokens": n_in, "new_tokens": len(reference)}
            for name, kwargs in methods.items():
                forwards[0] = 0
                tokens = model.generate(**inputs, **gen, **kwargs)[0, n_in:].tolist()[: args.max_new_tokens]
                exact[name] += tokens == reference
                row[name] = {"exact": tokens == reference, "forwards": forwards[0]}
        print(json.dumps(row), flush=True)
    for name, count in exact.items():
        print(f"{name}: exact {count}/{len(indices)}")


if __name__ == "__main__":
    main()
