"""actlib smoke test: one command to confirm the harness works.

Run:
    .venv/bin/python tools/actlib/smoke.py

Loads gpt2 (CPU), captures resid_post at 2 layers on 4 prompts, prints shapes,
ranks channels three ways, and runs an identity patch. First run downloads gpt2
to the HF cache; subsequent runs are offline.
"""

import os
import sys

# Make `import actlib` work when run as a script from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from actlib import (
    cache_site,
    capture_activations,
    extract_norm_gains,
    get_model_info,
    load_model,
    patch_and_run,
    rank_channels,
)


def main():
    model, tok = load_model("gpt2", device="cpu")
    info = get_model_info(model)
    print(f"model: gpt2  family={info.family}  layers={info.n_layers}  "
          f"hidden={info.hidden_size}  norm={info.norm_type}")

    prompts = [
        "The capital of France is",
        "Water boils at a temperature of",
        "The opposite of hot is",
        "Two plus two equals",
    ]
    layers = [4, 8]
    res = capture_activations(model, tok, prompts, sites="resid_post",
                              layers=layers, positions="last", batch_size=4)
    for layer in layers:
        print(f"resid_post L{layer}: {tuple(res[(layer, 'resid_post')].shape)}")

    acts = res[(8, "resid_post")]  # [4, hidden]
    gains = extract_norm_gains(model)["gains"][8]  # gain feeding block 8's norm
    for method, kwargs in [
        ("raw_magnitude", {}),
        ("abs_zscore", {}),
        ("gain_weighted_zscore", {"gains": gains}),
    ]:
        r = rank_channels(acts, method, exclude_sinks=True, top_k=5, **kwargs)
        print(f"top-5 {method:>21}: {r['indices'].tolist()}  "
              f"(excluded {r['excluded'].numel()} sinks)")

    # Identity patch: patch a prompt's own cached activation back in.
    prompt = prompts[0]
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        base = model(**enc).logits[0, -1]
    cached = cache_site(model, tok, prompt, "resid_post", 8)
    out = patch_and_run(model, tok, prompt, [cached])
    max_abs = (base - out["next_token_logits"]).abs().max().item()
    print(f"identity patch max |Δ logit| = {max_abs:.2e} "
          f"({'OK' if max_abs < 1e-2 else 'LARGE'})")
    print("smoke OK")


if __name__ == "__main__":
    main()
