"""Profiling + dev-set vanilla dumps (feeds bench_tree_spec/simulate.py).

modal run bench_tree_spec/run_prep.py
"""

import json
from pathlib import Path

from modal_app import app, dump_vanilla, profile


@app.local_entrypoint()
def prep(model: str = "Qwen/Qwen3-8B", n: int = 30, offset: int = 30, max_new_tokens: int = 384):
    out = Path("bench_tree_spec/results")
    out.mkdir(parents=True, exist_ok=True)
    prof_call = profile.spawn(model)
    dumps = list(
        dump_vanilla.starmap(
            [(t, n, offset, model, max_new_tokens) for t in ("code_edit", "humaneval", "gsm8k", "chat")]
        )
    )
    (out / "dev_vanilla.json").write_text(json.dumps(dumps))
    report = prof_call.get()
    (out / "profile.json").write_text(json.dumps(report))
    print("done")
