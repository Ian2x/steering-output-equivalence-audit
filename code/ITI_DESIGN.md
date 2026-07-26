# ITI arm — design note (Inference-Time Intervention, Li et al. 2306.03341)

**Status: LAUNCH-PREP (2026-07-07). Do NOT launch until the lead resolves the flagged
decisions in §7.** This note accompanies `exp/iti_steer.py` + `exp/run_iti.py`, drafted to
mirror the CAA arm (`caa_steer.py` / `run_caa.py`) under the frozen pre-registered battery
(`plan.md` §2–5, §8, §11; Amendments 1–3). Model: **Qwen/Qwen2.5-7B-Instruct** bf16 (the same
7B model the CAA arm runs on), to keep the 7B tier on one open, ungated checkpoint.

---

## 1. What `plan.md` / `references.md` pre-registered for ITI

- **§8 prediction table:** `ITI | head-shift | Llama-2-7b / baked | moderate; True×Info
  dissociation exploratory | **Mixed****. Predicted class = **Mixed**. κ moderate.
- **§2 control family:** ITI is listed under the *"Projection / all-position family
  (refusal-direction ablation, **ITI head shifts applied at all positions**)"*. Its
  pre-registered **primary control = calibrated logit bias tuned to the same behavior-rate
  shift on a calibration split (effect-space budget)**, evaluated on a **disjoint eval
  split**. Rationale given in-plan: "norm budgets are undefined for projections." (ITI is
  additive, not a projection — see the caveat in §4 and flag D-3 — but the plan explicitly
  files ITI in the effect-space-budget bucket, so we honor that.)
- **§7 native regime:** ITI is named as a natively **all-position** method → `E_native =
  E_all`, so κ = E_first / E_native is informative (same regime as CAA/refusal).
- **§16/Amendment-3:** confirms ITI ∈ natively-all-position set (E_native = E_all).
- **`references.md` line 47 (ITI target verification):** *"only head-output-site method
  (pre-W_O, **K=48 probe-selected heads, α=15**); baked HF 7B model. The 'TruthfulQA persona
  vs truth concept' critique is an output-vs-content question; **True×Info moves while MC
  barely does — a ready-made dissociation.**"* This pins the two ITI hyperparameters
  (**K = 48, α = 15**) and the injection site (**pre-W_O per-head activations**), both taken
  straight from Li et al.'s Llama-2-7B recipe.

**Frozen verdict rule (plan §3 as amended §11):** Dissolved = ρ_lo ≥ 0.9 on a *valid* cell;
Genuine = ρ_hi ≤ 0.3 with steering effect ≥ 3× floor (+ Amendment-2 dose-response if the
control voids); else Mixed. Verdict from **bootstrap-CI bounds + cell_valid**, never the ρ
point estimate. Cell valid ⇔ effect ≥ 3× floor AND E_native gate-clean AND control
gate-clean. Mandatory degeneracy guard (3-gram rep / median length / NLL) voids any cell.

---

## 2. ITI mechanics (faithful to Li et al. 2306.03341, `honest_llama` reference)

ITI is the only method in the audit that operates on **per-attention-head activations at the
head-output site (the input to W_O / `o_proj`, i.e. the concatenated per-head "z" before the
output projection mixes them)**. Faithful pipeline:

1. **Head activation extraction.** For a contrastive behavior dataset (statements labeled
   truthful vs untruthful, in Li et al.), run the model over each example and read the
   **per-head z at the last token** — the slice `h*head_dim:(h+1)*head_dim` of the `o_proj`
   *input* at every (layer L, head H). For Qwen2.5-7B: 28 layers × 28 heads × head_dim 128
   (hidden 3584, 28 query heads; GQA's 4 KV heads do **not** change the o_proj input width,
   which is always `n_query_heads*head_dim = hidden` — head-slicing the o_proj input is valid
   under GQA). This is the same head-slice trick `fv_extract.py` uses on Pythia's
   `attention.dense` input, ported to `self_attn.o_proj`.

2. **Per-head linear probes.** For each (L, H) train a logistic-regression probe on the
   head's z (label = behavior class) on a train split; record **validation accuracy**. The
   probe weight (unit-normalized) is that head's "truthful direction" θ_{L,H}.

3. **Top-K head selection.** Rank all L×H heads by validation accuracy; keep the **top K =
   48** (Li et al.'s Llama-2-7B setting).

4. **Inference-time intervention.** During generation, for each selected head add
   **α · σ_{L,H} · θ_{L,H}** to that head's z slice at **every position** (all-position, Li
   et al.'s deployment). σ_{L,H} = std of the head's activation *along θ* on the training set
   (the "along the direction" scaling that makes α dimensionless). **α = 15** (Li et al.).
   The shift is applied in the o_proj input, then o_proj/W_O mixes it into the residual
   stream as usual. Li et al. use the probe direction × σ; the released code also offers a
   "mass-mean" direction (μ_truthful − μ_untruthful projected). We implement the **probe
   direction** as primary (flag D-2 exposes mass-mean as an alternative).

**Why this is a distinct arm from CAA/refusal:** CAA adds one residual-stream vector at one
layer; refusal projects out one residual direction at all layers. ITI writes **K = 48
independent small per-head vectors at the head-output interface** — a distributed,
mid-network, additive edit. It is the audit's cleanest test of whether a *distributed
head-level probe-derived* intervention is still nothing but an output push.

---

## 3. Mapping onto native / first / base modes

Implemented in `iti_steer.py::ITIMethod`, structurally identical to `CAAMethod` (manual
generation loop so E_first can stop after step 0; KV-baked E_first verified by
`kv_baked_first_sanity`, mirroring `caa_steer.py`).

- **`base`** — no intervention, greedy. (= `B.base_generate`.)
- **`native` / `all`** — add α·σ·θ to every selected head's z slice at **every position**
  for the whole generation (Li et al.'s published deployment; E_native = E_all since ITI is
  natively all-position).
- **`first`** — apply the head shifts only while processing the prompt + the **first**
  generated token, baked into the KV cache, then removed (E_first). Because the intervention
  is at the attention head-output site (feeds the residual stream, hence the KV of later
  layers/positions), the prompt+first-token edit persists through the cache exactly as in the
  CAA/refusal E_first construction. κ = E_first / E_native.

The head-shift is realized with a **forward-pre-hook on each selected head's `o_proj`** that
edits the incoming z slice (same mechanism as `fv_extract._patch_one_head`, but *adding* a
fixed vector rather than *replacing* the slice, and applied at the requested positions rather
than only the last token). A per-layer hook handles all selected heads in that layer in one
pass. A position gate (`state["active"]`) toggles the shift live for the E_first cutoff.

---

## 4. Matched-budget output-push control + degeneracy guard

**Primary control (per plan §2, projection/all-position family): effect-space budget.**
- Discover the control token set exactly as the other additive/all-position arms do:
  regress E_native's **position-1 logit delta** onto W_U, take the smallest token set
  capturing 90 % of ‖delta‖² (cap 100) via `B.discover_token_set`. (`iti_steer.
  position1_logit_delta` computes the mean pos-1 logit delta of the all-position ITI shift.)
- **Budget = same behavior-rate shift on the calibration split** (effect-space), NOT TF-KL.
  This is the pre-registered projection/all-position budget. We bisect the logit-bias scalar
  on the calib split so the **control's behavior-rate gain matches E_native's behavior-rate
  gain** (helper `calibrate_bias_scalar_ratematch` added in `run_iti.py`, reusing
  `B.LogitBiasProcessor` + `B.control_generate` + the arm's behavior classifier). The
  refusal arm (same family) used effect-space budget; we follow it, not the CAA TF-KL path.
  - **Amendment-1 TF-KL is additionally reported as a sensitivity control** (reusing the CAA
    TF-KL calibration machinery, `iti_steer.teacher_forced_stepkl_native` +
    `B.calibrate_bias_scalar_stepkl`) so the arm carries both budgets, but the **headline ρ
    uses the pre-registered effect-space budget**. Flag D-4.
- **Floor (every cell):** the plan's projection-family floor is a "random-direction
  projection." ITI is additive, so the faithful matched floor is **K random head directions
  at matched per-head norm** (α·σ along a random unit vector in each selected head's slice),
  3 seeds — the additive analogue, and the honest floor for an additive head write. This
  mirrors the CAA random-matched-norm ×3 floor. Flag D-3 (floor form follows from the
  additive-vs-projection classification question).
- **ρ = E(primary control) / E(E_native)**, bootstrap 95 % CI over the eval split; **κ =
  E_first / E_native**, bootstrap CI. Both via `B.bootstrap_ratio_ci` (paired, as in CAA).

**Degeneracy guard (plan §4, mandatory).** Identical to CAA: per-condition `B.degeneracy_gate`
vs this-model baseline refs; `degenerate` = rep-collapse or length-collapse (NLL-only trip on
coherent chat is treated as a baseline artifact, per §4 and the CAA arm). A degenerate control
can never count as reproduction (Amendment 2) → if the effect-space control degenerates at the
rate-matched budget, run the **Amendment-2 dose-response** (≥3 sub-degenerate scales, each with
effect ≤ 0.3·E_native) before any verdict, exactly as `run_caa.py` does.

---

## 5. Behavior + classifier (see flag D-1)

Two candidate behaviors:
- **(A) Truthfulness / TruthfulQA** — ITI's *native* task. Faithful, matches Li et al.
  exactly, and directly enables the pre-registered **True×Info-vs-MC dissociation**
  (references.md line 47). Cost: needs a truthfulness judge; on a 7B, generation-level truth
  scoring without GPT-judge is noisy (the field uses a fine-tuned GPT-judge). We would score
  with a phrase/consistency proxy (as the other arms score with phrase classifiers) and
  clearly label it a proxy, OR use TruthfulQA-MC logprob margin as the effect metric.
- **(B) Sycophancy** — what the CAA arm used. Lets ITI reuse CAA's exact dataset + sycophancy
  classifier + eval prompts, making CAA vs ITI a **clean same-behavior, same-model,
  different-mechanism** comparison (residual-add vs head-shift). Cost: sycophancy is *not*
  ITI's native task; we would be applying the ITI machinery to a behavior it was not designed
  for (still legitimate — the audit asks whether the *mechanism* is an output push — but it
  forfeits the True×Info dissociation and departs from Li et al.).

**Both are defensible; they answer different questions.** The code is written **behavior-
agnostic**: a `BEHAVIOR` constant switches between `"truthfulqa"` and `"sycophancy"` dataset
loaders + classifiers, defaulting to **`"truthfulqa"`** (ITI-native, keeps the pre-registered
dissociation available). The lead must confirm — see flag D-1.

---

## 6. Expected verdict and why

**Pre-registered: Mixed (plan §8).** Reasoning consistent with the emerging map (README: every
audited method so far has κ ≈ 1, methods separate on ρ):
- **κ likely ≈ 1** (moderate-to-high). Like CAA/refusal/task-vectors, the effect is expected to
  be established by the first generated token; E_first ≈ E_native.
- **ρ is the open coordinate.** ITI heads are selected by a probe that *classifies the concept*,
  and the shift is along that probe direction — this is more concept-aligned than ActAdd, so a
  pure logit-bias control may reproduce **some but not all** of the effect → ρ in the middle
  band → **Mixed**. If the probe directions turn out to be dominated by their unembedding
  projection (an output-token story), ρ could climb toward Dissolved; if the head shifts carry
  genuine upstream truthful content, ρ_hi ≤ 0.3 → Genuine. Mixed is the honest prior; the arm
  is designed to let the data move it either way.
- **Secondary dissociation (report-only, exploratory, if behavior = truthfulqa):** True×Info
  (generation) should move under ITI while MC-accuracy barely does — the plan's flagged
  "output-vs-content" signature. Reported alongside, never gating the verdict.

---

## 7. FLAGGED DESIGN DECISIONS — LEAD MUST RESOLVE BEFORE LAUNCH

These are points where `plan.md`/`references.md` are silent or ambiguous. Each is implemented
with the most defensible default (guarded behind a named flag/constant) so the code is
launch-ready, but the lead should confirm or override. **None is guessed silently — all are
surfaced here.**

- **D-1 — Behavior / dataset (HIGHEST PRIORITY).** ITI's native task is truthfulness
  (TruthfulQA); the CAA arm used sycophancy. `plan.md` §8 says "ITI … True×Info dissociation
  exploratory," implying **truthfulness**, but never states the ITI behavior as a frozen §8
  metric the way it does for other arms (§4 lists behavior metrics for ActAdd/refusal/FV, not
  ITI). **Default implemented: `BEHAVIOR="truthfulqa"`** (ITI-native, preserves the True×Info
  dissociation). Alternative `"sycophancy"` (reuses the CAA dataset/classifier for a clean
  same-behavior CAA-vs-ITI contrast) is fully wired behind the same constant. **Lead: pick
  one.** If truthfulqa, also confirm the effect metric: generation-phrase-proxy truth score
  (default) vs TruthfulQA-MC logprob margin (flag D-1b).

- **D-2 — Head direction: probe weight vs mass-mean.** Li et al. report both; their headline
  uses the probe direction, the released code defaults to mass-mean ("com" / center-of-mass)
  for the shift while still selecting heads by probe accuracy. **Default implemented: probe
  weight direction** (`DIRECTION_KIND="probe"`); `"mass_mean"` (μ_truthful − μ_untruthful,
  unit-normalized) is available. Lead: confirm probe-direction is the faithful choice for our
  audit, or switch to mass-mean to match the reference code's *shift* exactly.

- **D-3 — Matched-budget & floor granularity for K heads (additive vs projection tension).**
  `plan.md` §2 files ITI in the *projection/all-position* family (→ effect-space budget,
  random-**projection** floor), but ITI is mechanically **additive** (a shift, not a
  projection). Decisions taken: **(a)** headline budget = **effect-space (behavior-rate
  match)** per the plan's explicit filing; **(b)** floor = **K random head directions at
  matched per-head α·σ norm** (the additive analogue, since a "random projection" is
  ill-defined for an additive head write) — 3 seeds. **(c)** "Matched budget" for the control
  is defined at the **aggregate behavior-rate** level (one calibrated logit-bias scalar
  matching E_native's rate gain), NOT per-head — the control is a single prompt-independent
  logit bias (plan §2 anti-cheat: one fixed control per arm). Lead: confirm the additive floor
  and effect-space budget are the intended reading of ITI's family placement, or direct a
  strict projection-family treatment.

- **D-4 — Which budget is the headline ρ.** Default: **effect-space (rate-matched)** ρ is the
  headline (pre-registered projection/all-position budget); Amendment-1 **TF-KL** ρ is reported
  as a sensitivity. The CAA arm's headline was TF-KL (additive family). Because ITI straddles
  the two families, the code computes **both** and labels effect-space as headline. Lead:
  confirm, or elevate TF-KL to headline if ITI is to be treated as additive-family throughout.

- **D-5 — Probe/selection split hygiene & K, α, layer band.** K = 48 and α = 15 come from
  `references.md` (Li et al. Llama-2-7B); Qwen2.5-7B has the *same* 28×28 = 784 head grid as
  Llama-2-7B (32×32 = 1024 there — **NOT identical**; Qwen2.5-7B is 28L×28H). K = 48 out of 784
  is a slightly larger fraction than 48/1024, but K, α are exposed as `--top-k-heads` /
  `--alphas` and swept in stage 1 with the reproduction gate, so the launch picks the
  (K, α[, layer-band]) that actually reproduces the behavior on *this* model rather than
  assuming the Llama numbers transfer. Default sweep: K ∈ {24, 48, 96}, α ∈ {5, 10, 15, 20}.
  Probe train/val split is disjoint from the held-out eval prompts (extraction pairs vs eval
  prompts split like CAA). Lead: confirm the sweep grid, or freeze K = 48 / α = 15 as
  confirmatory (no sweep) to match Li et al. exactly.

- **D-6 — Reproduction-gate threshold & metric.** CAA used ≥ +25 pts clean behavior gain.
  Default: same **+25 pts** on the ITI behavior (`--repro-threshold 25`). For truthfulness the
  natural gain may be smaller than sycophancy; if the lead picks truthfulqa, consider a lower
  threshold or the MC-margin gate (flag D-1b). Implemented default = 25 pts on the chosen
  behavior classifier.

---

## 8. Interfaces (both files `python3 -m py_compile`-clean)

`exp/iti_steer.py` (mirrors `caa_steer.py`):
- `build_chat`, model-load via `battery.load_model` (Qwen2.5-7B-Instruct path, identical to
  CAA).
- `head_z_activations(...)` — per-head z at last token (o_proj-input head slices), all L×H.
- `train_head_probes(...)` — per-head logistic probe; returns accuracy grid, unit directions
  θ, and σ (std along θ) grid.
- `select_top_heads(acc, K)` — top-K (layer, head, acc).
- `ITIMethod` — `generate(prompt, alpha, mode)` for base/native/first with the per-head
  forward-pre-hook shift and KV-baked E_first; `generate_with_fixed_head_vectors(...)` for the
  floor / alternative-direction controls.
- Battery helpers matching CAA: `position1_logit_delta`, `first_token_flip_count`,
  `teacher_forced_stepkl_native`, `kv_baked_first_sanity`, plus the behavior classifier(s)
  (`is_truthful`/`truth_rate` proxy and re-exported sycophancy classifier from `caa_steer`)
  and W_U behavior-span geometry (report-only).

`exp/run_iti.py` (mirrors `run_caa.py`):
- argparse flags matching CAA, adapted: `--device --dtype --n --n-calib --n-extract --tokens
  --n-boot --sweep-layers --top-k-heads --alphas --repro-threshold --stage --smoke --outdir`.
  (`--top-k-heads` replaces CAA's implicit single layer; `--alphas` replaces `--coeffs`;
  `--sweep-layers` restricts the head-probe layer band, default all.)
- stage1: (K, α[, layer-band]) sweep + ≥ +25 pts reproduction gate (disk-staged `stage.json`,
  resumable); stage2: battery reusing `exp/battery.py` (effect-space rate-matched control =
  headline ρ, TF-KL control = sensitivity, K-random-head floor ×3, verdict from CI bounds +
  cell_valid, Amendment-2 dose-response if the control voids); writes `results_full.json` +
  `report.md` incrementally (courier-safe, per the CAA ops lesson).
- AWS/staging conventions honored: `--outdir` default under
  `runs/steering-content-audit/2026-07-08-iti-7b`, `--smoke`, `--stage all`, resumable
  `stage.json`. S3 result prefix at launch must be `projects/steering-content-audit/runs/…`
  (instance-role `projects/*`-only scope, per log 2026-07-07). DLAMI python `/opt/pytorch/bin/
  python`. **This note and the code do NOT launch anything.**
