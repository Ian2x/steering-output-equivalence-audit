"""SAE feature-steering arm driver — Templeton-style feature steering on
GPT-2-small with a RELEASED open SAE (Joseph Bloom gpt2-small-res-jb,
blocks.7.hook_resid_pre), under the frozen pre-registered battery (plan.md
sections 2-5, 8, 11).

Method (faithful SAE feature steering)
--------------------------------------
1. Discover a clean interpretable concept feature programmatically (activation
   gap concept-vs-neutral); report top max-activating tokens as evidence.
2. Steering = add c * W_dec[f] (unit decoder direction) at resid_pre of the SAE
   layer, at EVERY position (native all-position SAE-steering deployment). The
   "clamp to k x max" variant is an equivalent scaled add; we report c.
3. Reproduction gate: sweep c for >= +30 pts clean concept-rate gain on the
   50-prompt calib split; pick highest clean gain. If no feature drives the
   behavior cleanly, try alternatives; else report "SAE steering did not
   reproduce" with the sweep (do not battery a non-effect).
4. Battery (150 neutral eval prompts reused from ActAdd/A0; 64 tok greedy; 10k
   bootstrap): baseline; E_native (all-position add); E_first (KV-baked
   prompt+first-tok; kappa = E_first/E_native, a NATIVELY-all-position method so
   kappa is informative, like refusal); primary control = calibrated static logit
   bias on regression-discovered token set, TF-per-step-KL budget matched to
   E_native (Amendment 1), prompt-independent; floor = random-direction
   matched-norm add, 3 seeds; W_U-concept-span secondary (report-only).

Verdict (plan section 3 + Amendment 1): Dissolved = rho_lo >= 0.9 on a valid cell
(effect >= 3x floor, gate clean); Genuine = rho_hi <= 0.3 with effect >= 3x floor
(+ Amendment-2 dose-response if control void); else Mixed. Verdict from CI bounds
+ cell_valid, NOT rho point estimate. rho/kappa = plain E-ratios.

Disk-staged (stage.json) so a timeout resumes without recompute. Foreground.

Usage:
  python run_sae.py --smoke --n 12 --tokens 24    # smoke (SAE load+recon+feature)
  python run_sae.py                                # full arm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import battery as B  # noqa: E402
import sae_steer as S  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
RUN_DIR = os.path.join(_REPO, "runs/steering-content-audit/2026-07-07-sae-arm")

SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.7.hook_resid_pre"

# Concept probe corpora for feature discovery (concept = wedding, for direct
# comparability with the ActAdd arm). Deterministic, fixed.
CONCEPT_TEXTS = [
    "The bride and groom exchanged vows at their wedding.",
    "We danced all night at the wedding reception.",
    "She wore a beautiful bridal gown to the marriage ceremony.",
    "Their honeymoon after the wedding was wonderful.",
    "The engagement led to a lovely wedding next spring.",
    "The bridesmaids gathered around the bride before the ceremony.",
    "They got married at the altar after a long engagement.",
    "The wedding invitations and bouquet were beautiful.",
]
NEUTRAL_TEXTS = [
    "The weather today is cloudy with a chance of rain.",
    "I spent the afternoon reading a book about history.",
    "The stock market rose slightly on Tuesday.",
    "He repaired the engine of the old car.",
    "The recipe calls for two cups of flour and sugar.",
    "The train departed from the station right on time.",
    "She planted tomatoes and herbs in the garden.",
    "The lecture covered the basics of thermodynamics.",
]
# Broader corpus for max-activating-token evidence.
MAXACT_CORPUS = CONCEPT_TEXTS + NEUTRAL_TEXTS + [
    "The bride and groom stood at the altar for the nuptials.",
    "Bridal showers and engagement parties precede the wedding.",
]


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def stage_path(outdir):
    return os.path.join(outdir, "stage.json")


def load_stage(outdir):
    p = stage_path(outdir)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"sweep": {}, "battery": {}}


def save_stage(outdir, stage):
    with open(stage_path(outdir), "w") as f:
        json.dump(stage, f, indent=2)


def gen_condition(fn, prompts):
    texts = [fn(p) for p in prompts]
    hits = [int(B.wedding_topic_hit(t)) for t in texts]
    return texts, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-calib", type=int, default=50)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default=RUN_DIR)
    ap.add_argument("--coeff-grid", default="20,40,60,80,100,140,180,240,320")
    ap.add_argument("--repro-threshold", type=float, default=30.0)
    ap.add_argument("--fixed-feature", type=int, default=None)
    ap.add_argument("--fixed-coeff", type=float, default=None)
    args = ap.parse_args()

    if args.smoke:
        args.n_boot = min(args.n_boot, 500)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    t0 = time.time()
    log(f"SAE arm: model={args.model} device={args.device} n={args.n} "
        f"tokens={args.tokens} tag={tag}")

    # --- prompts: reuse frozen A0/ActAdd neutral set + splits ---
    with open(os.path.join(_HERE, "prompts_neutral.json")) as f:
        prompts = json.load(f)["prompts"]
    assert len(prompts) == 200
    calib_all, eval_all = B.split_prompts(prompts, args.n_calib)
    if args.n < 200:
        n_eval = args.n - min(args.n_calib, args.n // 4)
        n_cal = args.n - n_eval
        calib, eval_ = calib_all[:n_cal], eval_all[:n_eval]
    else:
        calib, eval_ = calib_all, eval_all
    log(f"calib={len(calib)} eval={len(eval_)}")

    # --- model + SAE ---
    model, tok = B.load_model(args.model, device=args.device)
    device = args.device
    H = B.get_model_info(model).hidden_size
    n_layers = B.get_model_info(model).n_layers
    sae, sae_meta = S.load_sae(SAE_RELEASE, SAE_ID, device=device)
    L, site = S.sae_layer_from_hook(sae_meta["hook_name"])
    log(f"model n_layers={n_layers} hidden={H}; SAE {SAE_RELEASE}/{SAE_ID} "
        f"-> layer {L} site {site}, d_sae={sae_meta['d_sae']}")

    # --- reconstruction gate (must be >0.9 skip-pos0 to use the SAE) ---
    recon = S.reconstruction_cosine(sae, model, tok, CONCEPT_TEXTS + NEUTRAL_TEXTS,
                                    L, site, device)
    log(f"recon cosine: all={recon['mean_cosine_all_positions']:.4f} "
        f"skip_pos0={recon['mean_cosine_skip_pos0']:.4f} "
        f"(n_tok={recon['n_tokens']})")
    assert recon["mean_cosine_skip_pos0"] > 0.9, "SAE reconstruction gate failed"

    # --- feature discovery ---
    if args.fixed_feature is not None:
        feature = args.fixed_feature
        disc = {"winner": feature, "shortlist": [],
                "note": "FIXED via --fixed-feature"}
    else:
        disc = S.discover_concept_feature(sae, model, tok, CONCEPT_TEXTS,
                                          NEUTRAL_TEXTS, L, site, device, topk=8)
        feature = disc["winner"]
    direction = sae.W_dec[feature].detach().to("cpu").float()
    dir_norm = float(direction.norm())
    maxact = S.max_activating_tokens(sae, model, tok, feature, MAXACT_CORPUS,
                                     L, site, device, topn=15)
    log(f"chosen feature f={feature} (W_dec norm {dir_norm:.4f}); "
        f"top tokens: {[m['token'] for m in maxact[:8]]}")

    # --- geometry: cos(W_dec[f], W_U wedding span) ---
    wu_ids, wu_kept = B.resolve_a0_token_ids(tok)
    wu_basis = B.wu_wedding_span_basis(model, wu_ids).to(device)
    dir_proj = B.project_onto(direction.to(device).unsqueeze(0),
                              wu_basis).squeeze(0).to("cpu")
    cos_dir_wu = (float(torch.nn.functional.cosine_similarity(
        direction.unsqueeze(0), dir_proj.unsqueeze(0)).item())
        if dir_proj.norm() > 0 else 0.0)
    log(f"cos(W_dec[f], W_U wedding span) = {cos_dir_wu:.4f}")

    meth = S.SAESteerMethod(model, tok, L, direction, device=device,
                            max_new_tokens=args.tokens)

    # --- calib baseline (gate refs + reproduction) ---
    log("baseline generations on calibration...")
    cal_base_texts = [B.base_generate(model, tok, p, args.tokens, device)
                      for p in calib]
    cal_base_rate = float(np.mean([int(B.wedding_topic_hit(t))
                                   for t in cal_base_texts]))
    base_rep = float(np.mean([B.three_gram_rep_rate(t, tok)
                              for t in cal_base_texts]))
    base_median_len = B.median_len_tokens(cal_base_texts, tok)
    base_nll = float(np.mean([B.mean_nll_under_model(model, tok, p, t, device)
                              for p, t in zip(calib, cal_base_texts)]))
    log(f"calib baseline: rate={cal_base_rate:.3f} rep={base_rep:.3f} "
        f"med_len={base_median_len:.1f} nll={base_nll:.3f}")

    stage = load_stage(outdir)

    # ==================================================================
    #  A. REPRODUCTION SWEEP (coeff grid; native all-position steering)
    # ==================================================================
    coeff_grid = [float(x) for x in args.coeff_grid.split(",")]
    if args.fixed_coeff is not None:
        coeff_grid = [args.fixed_coeff]

    def _sweep_key(c):
        # tag + calib size + tokens in the key so smoke and full never collide
        # (they share stage.json but compute cells on different splits/lengths).
        return f"{tag}|nc{len(calib)}|t{args.tokens}|f{feature}|c{c}"

    for c in coeff_grid:
        key = _sweep_key(c)
        if key in stage["sweep"]:
            rec = stage["sweep"][key]
            log(f"  [cached] c={c}: rate={rec['rate']:.3f} "
                f"gain={rec['gain_pts']:+.1f} gate={rec['gate_tripped']}")
            continue
        texts = [meth.generate(p, c, "native") for p in calib]
        hits = [int(B.wedding_topic_hit(t)) for t in texts]
        rate = float(np.mean(hits))
        gate = B.degeneracy_gate(texts, calib, base_rep, base_median_len,
                                 base_nll, model, tok, device=device)
        gain = (rate - cal_base_rate) * 100
        stage["sweep"][key] = {
            "feature": feature, "coeff": c, "rate": rate, "gain_pts": gain,
            "gate_tripped": bool(gate.tripped), "gate_reasons": gate.reasons,
            "rep": gate.rep_rate, "median_len": gate.median_len,
            "nll": gate.mean_nll,
        }
        save_stage(outdir, stage)
        log(f"  c={c}: rate={rate:.3f} gain={gain:+.1f}pts gate={gate.tripped} "
            f"{gate.reasons}")

    sweep_table = sorted(stage["sweep"].values(), key=lambda v: v["coeff"])
    # Coherence-constrained reproduction selection (plan §11 Amendment 3 as a
    # HARD constraint, not a footnote): the §4 rep gate is calibrated on GPT-2's
    # already-high baseline rep (~0.60), so it does not trip when a condition
    # collapses into single-token loops. "highest clean gain, tie->lowest c"
    # therefore selects INTO degeneracy (it prefers the saturating-but-repetitive
    # high-c cell). Instead require rep <= baseline*1.5 (Amendment-3 threshold)
    # AND gain >= threshold, then pick the LOWEST coefficient that qualifies (the
    # least-degenerate dose that reproduces the behavior). Falls back to the old
    # rule only if no cell satisfies the rep constraint (reported as such).
    rep_cap = 1.5 * base_rep
    eligible = [v for v in sweep_table
                if not v["gate_tripped"] and v["gain_pts"] >= args.repro_threshold]
    coherent = [v for v in eligible if v["rep"] <= rep_cap]
    if coherent:
        best = min(coherent, key=lambda v: v["coeff"])
        chosen = {"feature": best["feature"], "coeff": best["coeff"],
                  "note": f"lowest coeff with clean gain >= +{args.repro_threshold} "
                          f"pts AND rep {best['rep']:.3f} <= 1.5*baseline "
                          f"{rep_cap:.3f} (Amendment-3 coherence constraint); "
                          f"gain {best['gain_pts']:+.1f} pts"}
        log(f"REPRODUCED (coherence-constrained): f={feature} c={chosen['coeff']} "
            f"gain={best['gain_pts']:+.1f} rep={best['rep']:.3f}")
    elif eligible:
        # No coherent dose; fall back but flag it loudly.
        best = min(eligible, key=lambda v: v["coeff"])
        chosen = {"feature": best["feature"], "coeff": best["coeff"],
                  "note": f"WARNING no clean dose has rep <= 1.5*baseline "
                          f"({rep_cap:.3f}); reproduces only in a degenerate "
                          f"regime. Lowest-c eligible: c={best['coeff']}, "
                          f"rep={best['rep']:.3f}, gain {best['gain_pts']:+.1f} pts "
                          f"(Amendment-3: treat as effectively degenerate)."}
        log(f"REPRODUCED (DEGENERATE regime only): f={feature} c={chosen['coeff']} "
            f"rep={best['rep']:.3f}")
    else:
        chosen = None

    result = {
        "meta": {
            "arm": "SAE feature steering (Templeton-style) wedding topic",
            "model": args.model, "device": device, "tag": tag,
            "n_total": len(prompts), "n_calib": len(calib),
            "n_eval": len(eval_), "max_new_tokens": args.tokens,
            "n_boot": args.n_boot, "hidden": H, "n_layers": n_layers,
            "sae": sae_meta, "sae_layer": L, "sae_site": site,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "sae_reconstruction": recon,
        "feature_discovery": {
            "concept": "wedding", "method": "largest mean activation gap "
                       "(concept - neutral) over non-pos0 tokens",
            "chosen_feature": feature, "wdec_norm": dir_norm,
            "shortlist": disc.get("shortlist", []),
            "max_activating_tokens": maxact,
            "cos_wdec_wu_wedding_span": cos_dir_wu,
        },
        "calib_baseline": {"rate": cal_base_rate, "rep": base_rep,
                           "median_len": base_median_len, "nll": base_nll},
        "reproduction_sweep": {
            "coeff_grid": coeff_grid, "threshold_pts": args.repro_threshold,
            "table": sweep_table, "chosen": chosen,
        },
    }

    if chosen is None:
        log("SAE steering did NOT reproduce at any swept coeff. STOPPING.")
        result["verdict"] = {
            "class": "NOT-REPRODUCED",
            "reason": (f"No coeff reached +{args.repro_threshold} clean "
                       f"wedding-rate gain on calibration for feature "
                       f"{feature}; battery NOT run."),
        }
        _write(result, outdir, tag, args)
        log("DONE — verdict=NOT-REPRODUCED (battery skipped)")
        return

    # ==================================================================
    #  B. BATTERY at the reproduced config
    # ==================================================================
    coeff = chosen["coeff"]
    log(f"BATTERY: f={feature} c={coeff} L={L} site={site}")

    # --- E_first KV-baked sanity (2 prompts) ---
    sanity = S.kv_baked_first_sanity(meth, tok, calib[:2], coeff)
    log(f"E_first KV-baked sanity: all_match={sanity['all_match']}")

    # --- EVAL conditions ---
    log("EVAL: baseline...")
    base_texts, base_hits = gen_condition(
        lambda p: B.base_generate(model, tok, p, args.tokens, device), eval_)
    log("EVAL: E_native (all-position add)...")
    nat_texts, nat_hits = gen_condition(
        lambda p: meth.generate(p, coeff, "native"), eval_)
    log("EVAL: E_first (KV-baked prompt+first-tok)...")
    first_texts, first_hits = gen_condition(
        lambda p: meth.generate(p, coeff, "first"), eval_)

    # --- floor: random direction matched norm c*||dir|| (=c, dir unit), 3 seeds
    matched_norm = coeff * dir_norm
    log(f"EVAL: floor (random dir, matched norm {matched_norm:.2f}, 3 seeds)...")
    floor_runs = []
    for s in range(3):
        g = torch.Generator().manual_seed(3000 + s)
        rv = torch.randn(H, generator=g)
        rv = rv / rv.norm() * matched_norm
        ftexts = [meth.generate_with_fixed_vector(p, rv, "native") for p in eval_]
        fhits = [int(B.wedding_topic_hit(t)) for t in ftexts]
        fr = float(np.mean(fhits))
        floor_runs.append({"seed": 3000 + s, "rate": fr, "hits": fhits,
                           "texts": ftexts})
        log(f"  floor seed {3000+s}: rate={fr:.3f}")
    floor_max = max(floor_runs, key=lambda r: r["rate"])

    # --- I2 primary control: token set from E_native pos-1 logit deltas, budget
    #     = mean teacher-forced per-step KL of E_native (Amendment 1). ---
    log("I2 control: token-set discovery (E_native pos-1 logit delta)...")
    mean_delta = S.position1_logit_delta(meth, tok, calib, coeff)
    ctrl_token_ids = B.discover_token_set(mean_delta, coverage=0.90, cap=100)
    top_ctrl_tokens = [tok.decode([i]) for i in ctrl_token_ids[:15]]
    log(f"  control token set size={len(ctrl_token_ids)}; top {top_ctrl_tokens}")

    log("I2 control: unsteered continuations on calib (for TF-KL)...")
    calib_cont_ids = [B.base_generate_ids(model, tok, p, args.tokens, device)
                      for p in calib]
    log("I2 control: B* = mean teacher-forced per-step KL (E_native)...")
    target_kl = S.teacher_forced_stepkl_native(meth, tok, calib, calib_cont_ids,
                                               coeff)
    log(f"  B* (E_native TF per-step KL) = {target_kl:.5f}")
    c_scalar, achieved_kl = B.calibrate_bias_scalar_stepkl(
        model, tok, calib, calib_cont_ids, ctrl_token_ids, mean_delta,
        target_kl, device=device)
    log(f"  bias scalar c={c_scalar:.4f} achieved TF-KL={achieved_kl:.5f}")
    tid_t = torch.tensor(ctrl_token_ids)
    bias_vals = c_scalar * mean_delta[tid_t]
    processor = B.LogitBiasProcessor(ctrl_token_ids, bias_vals)

    log("EVAL: control (calibrated logit bias)...")
    ctrl_texts, ctrl_hits = gen_condition(
        lambda p: B.control_generate(model, tok, p, processor, args.tokens,
                                     device), eval_)

    # --- W_U secondary control (report-only): project dir onto W_U wedding span,
    #     re-norm to matched_norm, add at native (all) positions ---
    log("W_U secondary control (report-only)...")
    proj_norm = float(dir_proj.norm())
    if proj_norm > 0:
        wu_vec = dir_proj / proj_norm * matched_norm
        wu_texts, wu_hits = gen_condition(
            lambda p: meth.generate_with_fixed_vector(p, wu_vec, "native"), eval_)
        wu_rate = float(np.mean(wu_hits))
    else:
        wu_texts, wu_hits, wu_rate = [], [], 0.0
    log(f"  W_U-control rate={wu_rate:.3f}")

    # --- mechanism: first-token argmax flips under E_native ---
    log("Mechanism: first-token argmax flips (E_native, eval)...")
    n_flips, n_flip_prompts = S.first_token_flip_count(meth, tok, eval_, coeff)
    log(f"  first-token flips = {n_flips}/{n_flip_prompts}")

    # --- rates + bootstrap CIs ---
    def rate_ci(hits):
        return B.bootstrap_rate_ci(hits, args.n_boot, seed=7)

    r_base = rate_ci(base_hits)
    r_nat = rate_ci(nat_hits)
    r_first = rate_ci(first_hits)
    r_ctrl = rate_ci(ctrl_hits)
    r_floor = rate_ci(floor_max["hits"])
    r_wu = rate_ci(wu_hits) if wu_hits else (wu_rate, float("nan"), float("nan"))

    # kappa = E_first / E_native (natively-all-position method, like refusal)
    kappa = B.bootstrap_ratio_ci(first_hits, nat_hits, base_hits, args.n_boot,
                                 seed=11)
    # rho = E(control) / E(E_native)
    rho = B.bootstrap_ratio_ci(ctrl_hits, nat_hits, base_hits, args.n_boot,
                              seed=13)

    # --- degeneracy gate per eval condition ---
    log("degeneracy gate (eval conditions)...")
    ev_base_rep = float(np.mean([B.three_gram_rep_rate(t, tok) for t in base_texts]))
    ev_base_med = B.median_len_tokens(base_texts, tok)
    ev_base_nll = float(np.mean([B.mean_nll_under_model(model, tok, p, t, device)
                                 for p, t in zip(eval_, base_texts)]))
    gates = {}
    gate_conditions = {
        "E_native": (nat_texts, eval_),
        "E_first": (first_texts, eval_),
        "control": (ctrl_texts, eval_),
        "floor_max": (floor_max["texts"], eval_),
    }
    if wu_texts:
        gate_conditions["wu_secondary"] = (wu_texts, eval_)
    for name, (texts, pr) in gate_conditions.items():
        gg = B.degeneracy_gate(texts, pr, ev_base_rep, ev_base_med, ev_base_nll,
                               model, tok, device=device)
        gates[name] = {"tripped": gg.tripped, "rep": gg.rep_rate,
                       "median_len": gg.median_len, "nll": gg.mean_nll,
                       "reasons": gg.reasons}
        log(f"  {name}: tripped={gg.tripped} rep={gg.rep_rate:.3f} "
            f"med={gg.median_len:.1f} nll={gg.mean_nll:.3f}")

    e_native = r_nat[0] - r_base[0]
    e_first = r_first[0] - r_base[0]
    e_floor = r_floor[0] - r_base[0]
    effect_over_floor = e_native / max(e_floor, 1e-9)

    # --- Amendment 2: control void -> dose-response ---
    dose = None
    control_tripped = gates["control"]["tripped"]
    if control_tripped:
        log("Amendment 2: control tripped gate -> dose-response")
        dose = run_dose_response(
            model, tok, eval_, ctrl_token_ids, mean_delta, c_scalar,
            ev_base_rep, ev_base_med, ev_base_nll, e_native, r_base[0], args,
            device)

    # --- verdict (amended plan §3, from CI bounds + cell_valid) ---
    rho_lo, rho_hi = rho[1], rho[2]
    kappa_lo, kappa_hi = kappa[1], kappa[2]
    effect_ge_3x_floor = e_native >= 3 * e_floor
    gate_clean_native = not gates["E_native"]["tripped"]
    gate_clean_control = not gates["control"]["tripped"]
    cell_valid = effect_ge_3x_floor and gate_clean_native and gate_clean_control

    dose_ok = dose["passes_amendment2"] if dose is not None else None
    dissolved = (rho_lo >= 0.9) and cell_valid
    if control_tripped:
        genuine = (dose_ok is True) and effect_ge_3x_floor
    else:
        genuine = (rho_hi <= 0.3) and effect_ge_3x_floor
    verdict = "Dissolved" if dissolved else ("Genuine" if genuine else "Mixed")

    result.update({
        "chosen_config": chosen,
        "chosen_coeff": coeff,
        "eval_baseline_refs": {"rep": ev_base_rep, "median_len": ev_base_med,
                               "nll": ev_base_nll},
        "method_fidelity": {
            "feature": feature, "layer": L, "site": site,
            "direction": "W_dec[f] (unit decoder direction)",
            "wdec_norm": dir_norm,
            "injection_native": "c*W_dec[f] added at EVERY position "
                                "(prompt + each generated token)",
            "injection_first": "c*W_dec[f] added at prompt positions + first "
                               "generated token, KV-baked, then removed",
            "kv_baked_first_all_match": sanity["all_match"],
        },
        "kv_baked_sanity": sanity,
        "rates": {
            "baseline": {"rate": r_base[0], "ci_lo": r_base[1], "ci_hi": r_base[2]},
            "E_native": {"rate": r_nat[0], "ci_lo": r_nat[1], "ci_hi": r_nat[2]},
            "E_first": {"rate": r_first[0], "ci_lo": r_first[1], "ci_hi": r_first[2]},
            "control": {"rate": r_ctrl[0], "ci_lo": r_ctrl[1], "ci_hi": r_ctrl[2]},
            "floor_max": {"rate": r_floor[0], "ci_lo": r_floor[1], "ci_hi": r_floor[2]},
            "wu_secondary": {"rate": r_wu[0], "ci_lo": r_wu[1], "ci_hi": r_wu[2]},
        },
        "kappa": {"point": kappa[0], "ci_lo": kappa[1], "ci_hi": kappa[2],
                  "note": "kappa = E_first / E_native. SAE steering is natively "
                          "all-position, so E_native = E_all and kappa is the "
                          "cascade share (like refusal): high kappa => the "
                          "first-position push self-sustains; low kappa => "
                          "sustained all-position application is needed."},
        "rho": {"point": rho[0], "ci_lo": rho[1], "ci_hi": rho[2],
                "note": "rho = E(control) / E(E_native)."},
        "effect": {"E_native": e_native, "E_first": e_first, "E_floor": e_floor,
                   "effect_over_floor": effect_over_floor},
        "control_calibration": {
            "token_set_size": len(ctrl_token_ids), "token_ids": ctrl_token_ids,
            "top_tokens": top_ctrl_tokens,
            "budget": "mean_teacher_forced_per_step_KL_of_E_native (Amendment 1)",
            "B_star_target_kl": target_kl, "achieved_kl": achieved_kl,
            "bias_scalar": c_scalar,
        },
        "wu_secondary_control": {
            "kept_tokens": wu_kept, "cos_wdec_wu_span": cos_dir_wu,
            "proj_norm": proj_norm, "matched_norm": matched_norm, "rate": wu_rate,
            "note": "report-only descriptive secondary (raw W_U columns as resid "
                    "directions, naive pullback, no lens)",
        },
        "geometry": {"cos_wdec_wu_wedding_span": cos_dir_wu,
                     "wdec_norm": dir_norm, "matched_floor_norm": matched_norm},
        "mechanism_check": {"first_token_flips": n_flips,
                            "n_prompts": n_flip_prompts,
                            "note": "E_native first-token argmax flips vs baseline"},
        "floor_runs": [{"seed": r["seed"], "rate": r["rate"]} for r in floor_runs],
        "degeneracy_gates": gates,
        "dose_response": dose,
        "samples": {
            "baseline": base_texts[:3], "E_native": nat_texts[:3],
            "E_first": first_texts[:3], "control": ctrl_texts[:3],
            "floor_max": floor_max["texts"][:3],
            "wu_secondary": wu_texts[:3] if wu_texts else [],
        },
        "verdict": {
            "class": verdict,
            "rho_lo": rho_lo, "rho_hi": rho_hi,
            "kappa_lo": kappa_lo, "kappa_hi": kappa_hi,
            "cell_valid": bool(cell_valid),
            "effect_ge_3x_floor": bool(effect_ge_3x_floor),
            "gate_clean_E_native": bool(gate_clean_native),
            "gate_clean_control": bool(gate_clean_control),
            "control_tripped": bool(control_tripped),
            "dose_response_passes_amendment2": dose_ok,
            "dissolved_rule": "rho_lo>=0.9 on valid cell",
            "genuine_rule": ("rho_hi<=0.3 & effect>=3x floor (control clean); OR "
                             "effect>=3x floor & Amendment-2 dose-response passes "
                             "(control void)"),
        },
        "runtime_sec": time.time() - t0,
    })

    _write(result, outdir, tag, args)
    log(f"DONE in {result['runtime_sec']:.0f}s  verdict={verdict}")


def run_dose_response(model, tok, eval_, ctrl_token_ids, mean_delta, c_full,
                      ev_rep, ev_med, ev_nll, e_native, base_rate, args, device):
    """Amendment 2: primary control at >=3 sub-degenerate scales. Passes if
    effect <= 0.3*E(native) at every non-degenerate scale."""
    scales = [0.25, 0.5, 0.75]
    tid_t = torch.tensor(ctrl_token_ids)
    rows = []
    for frac in scales:
        c = frac * c_full
        bias_vals = c * mean_delta[tid_t]
        proc = B.LogitBiasProcessor(ctrl_token_ids, bias_vals)
        texts = [B.control_generate(model, tok, p, proc, args.tokens, device)
                 for p in eval_]
        hits = [int(B.wedding_topic_hit(t)) for t in texts]
        rate = float(np.mean(hits))
        gate = B.degeneracy_gate(texts, eval_, ev_rep, ev_med, ev_nll, model,
                                 tok, device=device)
        eff = rate - base_rate
        ratio = eff / max(e_native, 1e-9)
        rows.append({"frac": frac, "bias_scalar": c, "rate": rate,
                     "effect": eff, "effect_over_native": ratio,
                     "gate_tripped": gate.tripped, "gate_reasons": gate.reasons})
        log(f"  dose frac={frac} c={c:.3f}: rate={rate:.3f} "
            f"eff/native={ratio:.3f} gate={gate.tripped}")
    clean = [r for r in rows if not r["gate_tripped"]]
    passes = (len(clean) >= 3) and all(r["effect_over_native"] <= 0.3
                                       for r in clean)
    return {"scales": scales, "rows": rows, "n_clean": len(clean),
            "passes_amendment2": bool(passes),
            "rule": ">=3 sub-degenerate scales, effect<=0.3*E(native) at each"}


def _write(result, outdir, tag, args):
    jpath = os.path.join(outdir, f"results_{tag}.json")
    with open(jpath, "w") as f:
        json.dump(result, f, indent=2)
    log(f"wrote {jpath}")
    if not args.smoke:
        with open(os.path.join(outdir, "results_full.json"), "w") as f:
            json.dump(result, f, indent=2)
        write_report(result, os.path.join(outdir, "report.md"))
        log(f"wrote {os.path.join(outdir, 'report.md')}")


def _fmt_ci(d):
    if d.get("ci_lo") != d.get("ci_lo"):
        return f"{d['rate']*100:.1f}% [n/a]"
    return f"{d['rate']*100:.1f}% [{d['ci_lo']*100:.1f}, {d['ci_hi']*100:.1f}]"


def _append_repro_section(A, r):
    rs = r["reproduction_sweep"]
    A(f"## Method reproduction (anti-strawman gate)\n")
    A(f"Coeff grid {rs['coeff_grid']} on the {r['meta']['n_calib']}-prompt "
      f"calibration split (E_native all-position add, {r['meta']['max_new_tokens']} "
      f"tok greedy). Gate: highest CLEAN (gate-not-tripped) wedding rate with gain "
      f">= +{rs['threshold_pts']} pts over baseline. Calib baseline wedding rate = "
      f"{r['calib_baseline']['rate']*100:.1f}%.\n")
    ch = rs.get("chosen")
    if ch:
        A(f"**Chosen config:** feature=`{ch['feature']}`, c={ch['coeff']} — "
          f"{ch['note']}.\n")
    else:
        A(f"**No coeff reproduced** (no clean cell reached the threshold).\n")
    A(f"| feature | c | rate | gain (pts) | gate | rep | med_len | nll |")
    A(f"|--------:|--:|-----:|-----------:|:----:|----:|--------:|----:|")
    for t in rs["table"]:
        star = " **<-**" if (ch and t["coeff"] == ch["coeff"]) else ""
        A(f"| {t['feature']} | {t['coeff']} | {t['rate']*100:.1f}% | "
          f"{t['gain_pts']:+.1f}{star} | {'YES' if t['gate_tripped'] else 'no'} | "
          f"{t['rep']:.3f} | {t['median_len']:.1f} | {t['nll']:.3f} |")
    A("")


def write_report(r, path):
    m = r["meta"]
    lines = []
    A = lines.append
    A(f"# SAE feature-steering arm — wedding topic (steering-content-audit)\n")
    A(f"**Run:** {m['timestamp']}  ")
    A(f"**Method:** SAE feature steering (Templeton-style: clamp/add an SAE "
      f"feature's decoder direction), wedding topic.  ")
    sae = m["sae"]
    A(f"**SAE:** `{sae['release']}` / `{sae['sae_id']}` "
      f"(d_in {sae['d_in']}, d_sae {sae['d_sae']}, normalize="
      f"{sae['normalize_activations']}, decoder-row-norm "
      f"{sae['wdec_row_norm_mean']:.3f}), hook `{sae['hook_name']}` -> "
      f"actlib layer {m['sae_layer']} site `{m['sae_site']}`.  ")
    A(f"**Model:** `{m['model']}` ({m['n_layers']} layers, hidden {m['hidden']}), "
      f"device `{m['device']}`.  ")
    A(f"**Prompts:** {m['n_total']} neutral (calib {m['n_calib']} / eval "
      f"{m['n_eval']}), {m['max_new_tokens']} new tokens greedy, {m['n_boot']} "
      f"bootstrap resamples. Prompt set/splits reused from A0/ActAdd.\n")

    rec = r["sae_reconstruction"]
    A(f"**SAE reconstruction gate:** mean cosine(resid, decode(encode(resid))) = "
      f"{rec['mean_cosine_skip_pos0']:.4f} (skip pos-0 attention sink; "
      f"{rec['mean_cosine_all_positions']:.4f} incl. pos-0), n_tok={rec['n_tokens']} "
      f"— passes >0.9.\n")

    if r.get("verdict", {}).get("class") == "NOT-REPRODUCED":
        A(f"## VERDICT: SAE steering did NOT reproduce\n")
        A(f"{r['verdict']['reason']}\n")
        _append_feature_section(A, r)
        _append_repro_section(A, r)
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return

    v = r["verdict"]
    A(f"## VERDICT: **{v['class']}**\n")
    A(f"Pre-registered prediction (plan §8): **Mixed or Dissolved** — an SAE "
      f"decoder direction for a token-level concept feature is plausibly close to "
      f"an output push; a genuine mid-layer concept feature could be Genuine. "
      f"Amended rules (§3, §11): Dissolved = rho_lo >= 0.9 on a valid cell; "
      f"Genuine = rho_hi <= 0.3 with effect >= 3x floor (+ Amendment-2 "
      f"dose-response if control void); else Mixed.\n")
    A(f"- rho = E(control)/E(E_native) = **{r['rho']['point']:.3f}** "
      f"[{r['rho']['ci_lo']:.3f}, {r['rho']['ci_hi']:.3f}] "
      f"(rho_lo={v['rho_lo']:.3f}, rho_hi={v['rho_hi']:.3f})")
    A(f"- kappa = E_first/E_native = **{r['kappa']['point']:.3f}** "
      f"[{r['kappa']['ci_lo']:.3f}, {r['kappa']['ci_hi']:.3f}] "
      f"(cascade share; natively all-position => informative)")
    A(f"- cell valid = **{v['cell_valid']}** (effect>=3x floor: "
      f"{v['effect_ge_3x_floor']}; gate clean E_native: {v['gate_clean_E_native']}; "
      f"gate clean control: {v['gate_clean_control']})")
    if v.get("control_tripped"):
        A(f"- control tripped degeneracy gate; Amendment-2 dose-response passes = "
          f"**{v.get('dose_response_passes_amendment2')}**")
    A("")

    _append_feature_section(A, r)
    _append_repro_section(A, r)

    mf = r["method_fidelity"]
    A(f"## Method fidelity\n")
    A(f"- **Feature** f={mf['feature']} at layer {mf['layer']} `{mf['site']}`; "
      f"steer direction = {mf['direction']} (norm {mf['wdec_norm']:.4f}).")
    A(f"- **E_native** (published SAE-steering): {mf['injection_native']}.")
    A(f"- **E_first**: {mf['injection_first']}.")
    A(f"- **E_first KV-baked sanity:** reproduced by manual prefill-then-continue "
      f"(all_match={mf['kv_baked_first_all_match']}).\n")

    A(f"## Headline rates (eval split, {m['n_eval']} prompts)\n")
    A(f"Wedding-topic rate, bootstrap 95% CI in brackets.\n")
    rr = r["rates"]
    A(f"| condition | rate [95% CI] |")
    A(f"|---|---|")
    A(f"| baseline (unsteered) | {_fmt_ci(rr['baseline'])} |")
    A(f"| E_native (all-position add) | {_fmt_ci(rr['E_native'])} |")
    A(f"| E_first (KV-baked prompt+first-tok) | {_fmt_ci(rr['E_first'])} |")
    A(f"| control (calibrated logit bias) | {_fmt_ci(rr['control'])} |")
    A(f"| floor (random dir matched norm, max of 3) | {_fmt_ci(rr['floor_max'])} |")
    A(f"| W_U secondary control (report-only) | {_fmt_ci(rr['wu_secondary'])} |")
    A("")

    k = r["kappa"]; rho = r["rho"]; e = r["effect"]
    A(f"## Decomposition\n")
    A(f"- **kappa = E_first / E_native** = {k['point']:.3f} "
      f"[{k['ci_lo']:.3f}, {k['ci_hi']:.3f}]  — {k['note']}")
    A(f"- **rho = E(control) / E(E_native)** = {rho['point']:.3f} "
      f"[{rho['ci_lo']:.3f}, {rho['ci_hi']:.3f}]")
    A(f"- effect E_native = {e['E_native']*100:.1f} pts; E_first = "
      f"{e['E_first']*100:.1f} pts; floor = {e['E_floor']*100:.1f} pts; "
      f"E_native / floor = {e['effect_over_floor']:.2f}x (needs >= 3x).\n")

    cc = r["control_calibration"]
    A(f"## I2 primary control (Amendment 1: teacher-forced per-step KL of E_native)\n")
    A(f"- Token set S: {cc['token_set_size']} tokens (90% of ||E_native pos-1 "
      f"logit-delta||^2, cap 100). Top: {cc['top_tokens']}")
    A(f"- B* (E_native TF per-step KL) = {cc['B_star_target_kl']:.5f}, achieved "
      f"control TF per-step KL = {cc['achieved_kl']:.5f}, bias scalar = "
      f"{cc['bias_scalar']:.4f}.\n")

    if r.get("dose_response"):
        d = r["dose_response"]
        A(f"## Amendment 2 dose-response (control tripped gate)\n")
        A(f"Rule: {d['rule']}. Passes = **{d['passes_amendment2']}** "
          f"({d['n_clean']} clean scales).\n")
        A(f"| frac | bias scalar | rate | effect/native | gate |")
        A(f"|-----:|------------:|-----:|--------------:|:----:|")
        for row in d["rows"]:
            A(f"| {row['frac']} | {row['bias_scalar']:.3f} | {row['rate']*100:.1f}% "
              f"| {row['effect_over_native']:.3f} | "
              f"{'VOID' if row['gate_tripped'] else 'ok'} |")
        A("")

    wu = r["wu_secondary_control"]; g = r["geometry"]
    A(f"## Geometry + W_U secondary control (descriptive)\n")
    A(f"- **cos(W_dec[f], W_U-wedding-span projection)** = "
      f"{g['cos_wdec_wu_wedding_span']:.4f} (how much of the decoder direction "
      f"already lies in the naive unembedding span of the wedding lexicon).")
    A(f"- W_U secondary control (W_dec[f] projected onto span(W_U[wedding]), "
      f"re-normed to c*||W_dec[f]||={g['matched_floor_norm']:.1f}, added at all "
      f"positions): rate {wu['rate']*100:.1f}%. Report-only. Kept tokens: "
      f"{wu['kept_tokens']}.\n")

    mc = r["mechanism_check"]
    A(f"## Mechanism check\n")
    A(f"- First-token argmax flips vs baseline under E_native = "
      f"**{mc['first_token_flips']}/{mc['n_prompts']}**.\n")

    A(f"## Degeneracy gate (per eval condition)\n")
    er = r["eval_baseline_refs"]
    A(f"Eval baseline refs: rep={er['rep']:.3f}, median_len={er['median_len']:.1f}, "
      f"nll={er['nll']:.3f}. Gate: rep > 2x+0.1, or median_len < 0.5x, or nll > 3x.\n")
    A(f"| condition | tripped | rep | median_len | nll | reasons |")
    A(f"|---|:---:|---:|---:|---:|---|")
    for name, gg in r["degeneracy_gates"].items():
        A(f"| {name} | {'VOID' if gg['tripped'] else 'ok'} | {gg['rep']:.3f} | "
          f"{gg['median_len']:.1f} | {gg['nll']:.3f} | {'; '.join(gg['reasons'])} |")
    A("")

    A(f"## Floor runs\n")
    for fr in r["floor_runs"]:
        A(f"- seed {fr['seed']}: rate {fr['rate']*100:.1f}%")
    A("")

    A(f"## Sample generations (first 3 eval prompts per condition)\n")
    smp = r["samples"]
    for name in ["baseline", "E_native", "E_first", "control", "floor_max",
                 "wu_secondary"]:
        A(f"**{name}:**")
        for t in smp.get(name, []):
            A(f"  - {t!r}")
        A("")

    A(f"Runtime: {r['runtime_sec']:.0f}s.\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _append_feature_section(A, r):
    fd = r["feature_discovery"]
    A(f"## SAE feature discovery (programmatic, no Neuronpedia)\n")
    A(f"Concept = **{fd['concept']}** (for direct comparability with the ActAdd "
      f"arm). Method: {fd['method']}. Chosen feature = **{fd['chosen_feature']}** "
      f"(W_dec norm {fd['wdec_norm']:.4f}).\n")
    if fd.get("shortlist"):
        A(f"Top features by concept-neutral activation gap:\n")
        A(f"| feature | gap | concept act | neutral act |")
        A(f"|--------:|----:|------------:|------------:|")
        for s in fd["shortlist"]:
            star = " **<-**" if s["feature"] == fd["chosen_feature"] else ""
            A(f"| {s['feature']}{star} | {s['gap']:.3f} | {s['concept_act']:.3f} | "
              f"{s['neutral_act']:.3f} |")
        A("")
    A(f"**Feature {fd['chosen_feature']} top max-activating tokens** (evidence it "
      f"is the concept feature):\n")
    A(f"| token | activation |")
    A(f"|---|---:|")
    for mm in fd["max_activating_tokens"]:
        A(f"| {mm['token']!r} | {mm['act']:.2f} |")
    A("")
    A(f"cos(W_dec[f], W_U wedding span) = {fd['cos_wdec_wu_wedding_span']:.4f}.\n")


if __name__ == "__main__":
    main()
