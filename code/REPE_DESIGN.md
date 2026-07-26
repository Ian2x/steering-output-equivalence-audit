# RepE reading-vector arm — design note (Representation Engineering, Zou et al. 2310.01405)

**Status: LAUNCH-PREP (2026-07-07). Do NOT launch until the lead resolves the flagged
decisions in §7.** This note accompanies `exp/repe_steer.py` + `exp/run_repe.py`, drafted to
mirror the CAA arm (`caa_steer.py` / `run_caa.py`) as closely as possible under the frozen
pre-registered battery (`plan.md` §2–5, §8, §11; Amendments 1–3). Model:
**Qwen/Qwen2.5-7B-Instruct** bf16 — the *same* 7B model, behavior, dataset, classifier,
injection family, and read site/token as the CAA arm. **The one genuinely new component is the
direction extraction: PCA/LAT top component instead of CAA's mean difference.** Everything else
is CAA's, reused.

---

## 1. What `plan.md` / `references.md` pre-registered for RepE

Quoted verbatim (RepE is thinly specified in-plan; I quote everything I found and flag the
silences):

- **§8 prediction table (`plan.md:79`):**
  `| RepE reading/steering | additive | 7B tier | high κ, high ρ | Dissolved or Mixed |`.
  Predicted class = **Dissolved or Mixed**; κ **high**, ρ **high**; family = **additive**;
  tier = **7B**. This is the only per-method row for RepE.
- **Tier-B listing (`plan.md:64`):** *"CAA Llama-2-7b-chat, ITI baked Llama, **RepE
  reading-vector arm**, persona vectors."* Confirms the arm is the **reading-vector** operator
  at the **7B tier** (lead-fixed to Qwen2.5-7B-Instruct for one open, ungated 7B checkpoint,
  matching CAA/ITI).
- **`references.md:52`:** *"**RepE** (2310.01405, MIT) audited via **its shallowest
  contrast-vector operator first**."* The reading vector (LAT contrast direction) *is* that
  shallowest operator — this pins the operator choice and frames the high-ρ prediction (a
  contrast direction is the most likely to be an output push).
- **§2 control family (`plan.md:22` and the additive-family text):** the plan splits controls
  into an **additive family** (norm-budget / TF-KL logit bias) and a **projection /
  all-position family** (effect-space budget). **§8 files RepE as `additive`**, so RepE takes
  the **additive-family control = TF-KL-calibrated logit bias** (the CAA headline), NOT the
  projection/effect-space budget ITI used. This is unambiguous and makes RepE a *cleaner*
  drop-in on the CAA battery than ITI (which straddled the two families).
- **§7 native regime / Amendment 3:** the plan enumerates the natively-all-position set as
  "CAA, refusal ablation, ITI." **RepE is NOT explicitly named in that enumeration** — a
  silence (flag D-3). We default RepControl to **all-position deployment** (Zou et al.'s
  `RepControl` adds the reading vector at all token positions), so E_native = E_all and κ is
  informative exactly as for CAA. Flagged because the plan does not spell it out for RepE.
- **Silences noted explicitly:** the plan does **not** state (a) single- vs multi-layer
  injection for RepE, (b) PCA-top-1 vs LAT-signed vs stacked-reps extraction, (c) the
  contrastive stimulus template, or (d) a RepE-specific reproduction threshold. All are
  surfaced in §7 with the most-defensible default named and guarded.

**Frozen verdict rule (plan §3 as amended §11):** Dissolved = ρ_lo ≥ 0.9 on a *valid* cell;
Genuine = ρ_hi ≤ 0.3 with steering effect ≥ 3× floor (+ Amendment-2 dose-response if the
control voids); else Mixed. Verdict from **bootstrap-CI bounds + cell_valid**, never the ρ
point estimate. Cell valid ⇔ effect ≥ 3× floor AND E_native gate-clean AND control gate-clean.
Mandatory degeneracy guard (3-gram rep / median length / NLL) voids any cell.

---

## 2. RepE mechanics (faithful to Zou et al. 2310.01405, the LAT / reading-vector method)

RepE's reading-vector pipeline, as implemented in `repe_steer.py::read_vectors_pca`:

1. **Contrastive stimulus reps.** For each contrastive sycophancy pair (Rimsky A/B item;
   §5), render the chat prompt ending in the committed answer letter `"(A"` / `"(B"` and read
   **resid_post at the last (answer-letter) token** for both the sycophantic and the
   non-sycophantic letter. **This is the IDENTICAL read site and token position as
   `caa_steer.caa_vector`** — `repe_steer` literally calls CAA's private reader
   `C._last_token_resid_post`, so CAA and RepE read the *exact same activations*; only what is
   done with them differs.

2. **LAT paired differences.** Form `d_i = resid(sycophantic) − resid(non_syc)` over the
   dataset (Linear Artificial Tomography's contrastive-difference construction — the same
   contrastive pairing CAA averages). Flag D-6 exposes the *stacked-reps* alternative (PCA on
   the raw reps rather than the paired diffs).

3. **PCA top component.** (Optionally mean-center the diffs — flag D-5.) Run SVD on the
   (centered) `[n_pairs, hidden]` difference matrix and take the **top principal component** as
   the reading direction. `n_components = 1` is LAT's canonical reading vector (flag D-7 exposes
   `--n-components`). Numerical fallback to a covariance eigendecomposition if SVD fails.

4. **Sign alignment.** SVD/PCA fixes a direction only up to sign; a flipped sign would push
   *away* from sycophancy and spuriously fail the +25 gate. We align so **+v̂ pushes TOWARD
   sycophancy-positive**: the difference vectors point syc-minus-nonsyc, so we flip v̂ iff
   `mean_i(d_i · v̂) < 0`. We record `label_frac_aligned` (fraction of `d_i` on the +v̂ side)
   and `sign_align_mean_proj` for a launch-time audit (§7 D-4).

5. **Unit-normalize** v̂. Steering adds `c · v̂` at resid_post (coeff = added-vector norm),
   **byte-for-byte CAA's injection** — v̂ is unit in both arms, so the coefficient sweep is
   directly comparable and the CAA-vs-RepE contrast is *only* mean-diff-vs-PCA.

**Why this is a distinct arm from CAA:** CAA's direction is the *mean* of the contrastive
differences; RepE's is the *top principal component* (the leading axis of variation) of the same
differences. They coincide only if the differences are rank-1 (all parallel). The audit's
controlled variable is precisely this direction-derivation, holding model + behavior + injection
family + read site fixed. We report `cos(PCA reading vector, CAA mean-diff)` per layer so the
lead can see how far LAT departs from CAA before any behavior is measured.

**Reused from CAA (NOT reinvented):** the additive resid_post hook, the base/native/first
generation loop, the KV-baked E_first construction, `position1_logit_delta`,
`first_token_flip_count`, `teacher_forced_stepkl_native`, `kv_baked_first_sanity`, the sycophancy
classifier, the W_U-span geometry, the entire `battery.py` control/verdict stack, and the driver
scaffold (dataset loaders, split, +25 gate, incremental courier-safe writes, resumable
`stage.json`). The diff from `caa_steer.py` is centered on `read_vectors_pca` + the
`inject_layers` flag.

---

## 3. Mapping onto native / first / base modes

Implemented in `repe_steer.py::RepEMethod`, structurally identical to `CAAMethod` (manual
generation loop so E_first can stop after step 0; KV-baked E_first verified by
`kv_baked_first_sanity`, mirroring `caa_steer.py`). For `inject_layers="single"` the generation
path is a behavioral copy of `CAAMethod` (one layer, one vector).

- **`base`** — no intervention, greedy. (= the base branch of `RepEMethod._gen`, matching
  `CAAMethod`.)
- **`native` / `all`** — add `c·v̂` at resid_post at **every position** for the whole
  generation (Zou et al.'s RepControl deployment; E_native = E_all since RepControl is natively
  all-position). In `single` mode this is one layer; in `all` mode each active layer adds its
  **own** reading vector simultaneously.
- **`first`** — apply the add only while processing the prompt + the **first** generated token,
  baked into the KV cache, then removed (E_first). Because the edit is at resid_post (feeds the
  KV of later positions), the prompt+first-token edit persists through the cache exactly as in
  the CAA E_first construction. **κ = E_first / E_native**, CAA-comparable in `single` mode
  (see §7 D-1 for the multi-layer caveat).

The additive edit is realized with a **forward hook on each active layer's block** (resid_post),
adding the fixed per-layer vector at the requested positions, with a live `state["active"]`
toggle for the E_first cutoff — the same mechanism as `caa_steer._gen`, generalized over a set
of layers via an `ExitStack` of hooks. For `single` mode the set is one layer, so the mechanism
is identical to CAA.

---

## 4. Matched-budget output-push control + degeneracy guard

**Primary control (per plan §2, RepE is filed `additive` in §8): TF-KL logit-bias budget —
the CAA headline, reused verbatim.**
- Discover the control token set exactly as CAA does: `repe_steer.position1_logit_delta` computes
  E_native's **position-1 logit delta**; `B.discover_token_set` takes the smallest token set
  capturing 90 % of ‖delta‖² (cap 100).
- **Budget = mean teacher-forced per-step KL of E_native** (Amendment 1, TF-KL), via
  `repe_steer.teacher_forced_stepkl_native` → B\*, then `B.calibrate_bias_scalar_stepkl` bisects
  the logit-bias scalar so the control's TF-KL matches B\*. `B.LogitBiasProcessor` +
  `B.control_generate` apply it. **This is the CAA headline control unchanged** — no
  effect-space/rate-match path (that was ITI's projection-family treatment; RepE is additive, so
  it inherits CAA's budget directly and cleanly).
- **Floor (every cell):** random-direction, matched-norm add (`coeff·‖v̂‖ = coeff`, v̂ unit),
  **×3 seeds** — identical to CAA. `RepEMethod.generate_with_fixed_vector` supplies the random
  vector at the active layer(s)/positions/norm.
- **ρ = E(primary control) / E(E_native)**, bootstrap 95 % CI over the eval split; **κ =
  E_first / E_native**, bootstrap CI. Both via `B.bootstrap_ratio_ci` (paired, as in CAA).
- **W_U sycophancy-span secondary** (report-only): `cos(v̂, span(W_U[syc tokens]))` + a
  projected-and-re-normed steer, exactly as CAA reports it. Directly probes the §8 high-ρ
  prediction (is the reading vector mostly its unembedding shadow?).

**Degeneracy guard (plan §4, mandatory).** Identical to CAA: per-condition `B.degeneracy_gate`
vs this-model baseline refs; `degenerate` = rep-collapse or length-collapse (NLL-only trip on
coherent chat is a baseline artifact, per §4 and the CAA arm). A degenerate control can never
count as reproduction (Amendment 2) → if the control degenerates at the TF-KL budget, run the
**Amendment-2 dose-response** (≥3 sub-degenerate scales, each with effect ≤ 0.3·E_native)
before any verdict, exactly as `run_caa.py` does (`run_dose_response`).

---

## 5. Behavior + classifier

**Behavior = sycophancy — the CAA arm's dataset + classifier + eval prompts, reused verbatim.**
`repe_steer` re-exports `caa_steer`'s `is_sycophantic` / `sycophancy_rate` / `build_chat`; the
driver reuses `run_caa`'s Rimsky-CAA sycophancy loader (`nrimsky/CAA generate_dataset.json`, A/B
matched pairs, Anthropic Perez 2212.09251 fallback), the same held-out open-ended eval-prompt
construction, and the same extraction/eval split.

**Rationale (lead-fixed):** the audit's controlled variable is the direction-*derivation*
(mean-diff vs PCA vs per-head-probe), so behavior + model + injection-family are held fixed
across CAA/RepE/ITI for a clean three-way contrast. RepE's reading-vector extraction is
behavior-agnostic (PCA on *any* contrastive stimulus set), so sycophancy is a legitimate
application, and it makes **CAA-vs-RepE a same-behavior, same-model, same-read-site,
same-injection contrast in which literally only the direction estimator differs.** RepE's
*native* demo tasks are honesty / harmlessness; I considered whether honesty is materially
better *for the audit* and concluded **no** — switching behaviors would confound the
direction-derivation comparison with a behavior change and forfeit the clean CAA contrast (that
is the whole point of holding behavior fixed; contrast with ITI, whose native task genuinely
buys a True×Info dissociation). If the lead wants RepE's native honesty task instead, it is a
one-loader change, but it breaks the three-way contrast — see §7 D-2.

---

## 6. Expected verdict and why

**Pre-registered: Dissolved or Mixed (plan §8), κ high, ρ high.** Reasoning consistent with the
emerging map (README: every audited method so far has κ ≈ 1; methods separate on ρ):
- **κ likely ≈ 1** (high). Like CAA / refusal / task-vectors / SAE, the effect is expected to be
  established by the first generated token; E_first ≈ E_native. (In `all` mode κ is measured but
  is **not** CAA-comparable — §7 D-1.)
- **ρ is the open coordinate, and RepE is predicted HIGH-ρ.** references.md:52 calls the reading
  vector RepE's *shallowest* contrast operator, and a top-PCA contrast direction is, if anything,
  even more likely than CAA's mean-diff to lie in the unembedding span (the leading axis of the
  contrastive differences is plausibly dominated by the answer-token logit direction). So a pure
  TF-KL logit-bias control may reproduce **most** of the effect → ρ high → **Dissolved** (ρ_lo ≥
  0.9) or upper-**Mixed**. If the PCA direction instead captures genuine upstream sycophancy
  content beyond the mean-diff, ρ_hi could fall → Mixed/Genuine. **Dissolved-or-Mixed is the
  honest prior; the arm lets the data move it.**
- **CAA-vs-RepE diagnostic (report-only):** `cos(PCA reading vector, CAA mean-diff)` and the two
  arms' ρ side-by-side answer the actual scientific question — *does swapping the direction
  estimator (mean → top-PC) change where the method lands on the shallowness map?* If the cosine
  is ≈ 1 and ρ matches CAA's, the estimator choice is immaterial (a useful null); if they
  diverge, the audit has found that the *derivation* matters. Reported alongside, never gating
  the verdict.

---

## 7. FLAGGED DESIGN DECISIONS — LEAD MUST RESOLVE BEFORE LAUNCH

These are points where `plan.md`/`references.md` are silent or ambiguous. Each is implemented
with the most defensible default (guarded behind a named flag/constant) so the code is
launch-ready, but the lead should confirm or override. **None is guessed silently — all are
surfaced here.**

- **D-1 — Single-layer vs multi-layer injection (HIGHEST PRIORITY).** Zou et al. often add a
  reading vector at **every** layer (each layer's own vector); CAA adds at **one** layer.
  - **Single-layer** (`--inject-layers single`, **DEFAULT**) = a clean drop-in CAA contrast:
    same injection pattern, only the direction differs → E_native = E_all at one layer, and the
    κ machinery is *identical* to CAA. This is the faithful choice **for the audit**, whose
    controlled variable is the direction estimator — matching CAA's injection pattern isolates
    it. **Recommended default.**
  - **Multi-layer** (`--inject-layers all`) = more faithful to the *paper's* RepControl, but a
    **different injection pattern** that (a) muddies the CAA contrast (now two things differ:
    direction *and* layer-count), (b) complicates E_first/κ — κ in `all` mode is measured but is
    explicitly labeled **NOT CAA-comparable** in the results/report, and (c) makes the matched
    floor / TF-KL budget span many layers at once. Fully wired (each swept layer adds its own
    cached v̂ simultaneously; the sweep-over-L then only re-labels the κ/report layer and the
    driver short-circuits the redundant layer loop).
  - **Lead: confirm single-layer as the audit-faithful default, or request multi-layer as a
    paper-faithful companion run** (both can be launched; they answer different questions —
    "does the *direction* matter, injection held fixed" vs "does full RepControl land where CAA
    does"). My recommendation: **launch single-layer as the headline; optionally add a
    multi-layer companion for paper-fidelity, clearly marked non-CAA-comparable on κ.**

- **D-2 — Behavior/dataset: sycophancy (CAA-shared) vs RepE-native honesty.** Default
  implemented: **sycophancy**, reusing the CAA dataset/classifier/eval-prompts for a clean
  same-behavior CAA-vs-RepE contrast (the lead-fixed choice; §5). RepE's native tasks are
  honesty/harmlessness. Switching to honesty would confound the direction-derivation comparison
  with a behavior change and forfeit the three-way (CAA/RepE/ITI) contrast, so I did **not** do
  it. **Lead: confirm sycophancy** (recommended, preserves the contrast), or direct a native
  honesty run (one-loader change, breaks the contrast).

- **D-3 — Native position regime for RepE (all-position).** plan §7/Amendment-3 names CAA,
  refusal, ITI as natively-all-position but is **silent on RepE**. Zou et al.'s RepControl adds
  the reading vector at **all positions**, so I default RepE to **all-position** deployment
  (E_native = E_all, κ informative, same regime as CAA). **Lead: confirm RepE ∈
  natively-all-position set** (recommended, matches RepControl and the CAA regime), else specify
  a prompt-position-only regime (would make E_native ≈ E_first, κ ≈ 1 by construction, as for
  ActAdd).

- **D-4 — PCA sign-alignment convention (correctness-critical).** PCA/SVD fixes the reading
  direction only up to sign; a flipped sign pushes *away* from sycophancy and spuriously fails
  the +25 gate. **Default:** align +v̂ to the sign of `mean_i(d_i · v̂)` where `d_i =
  resid(syc) − resid(non_syc)` — i.e. +v̂ is the pro-sycophancy direction. The code records
  `label_frac_aligned` (should be ≳ 0.5; near 0.5 means the PCA axis barely separates the classes
  → the reading vector is weak and the sweep should show a small gain, a real signal not a bug)
  and `sign_align_mean_proj`. **Lead: confirm the sign convention;** at launch, eyeball
  `label_frac_aligned` per layer in `stage1_sweep.json` — a value < 0.5 after alignment would
  indicate a sign or extraction bug (see §10 risk R1).

- **D-5 — Mean-center the paired diffs before PCA.** LAT/PCA conventionally mean-centers.
  **Default:** mean-center on (`--no-mean-center` to disable). Centering removes the shared
  mean-difference component (exactly CAA's direction) so the top PC captures the leading *axis of
  variation* around it; NOT centering lets the top PC be pulled toward the mean-difference
  (making RepE ≈ CAA). Both are defensible; centering is the more standard LAT reading and the
  more informative contrast to CAA. **Lead: confirm mean-centering** (recommended), or request
  the uncentered variant if you want RepE to hew closer to CAA. (The `cos_to_meandiff` diagnostic
  makes the consequence visible either way.)

- **D-6 — Extraction template: paired-difference PCA vs stacked-reps PCA vs stimulus set.**
  **Default:** PCA on the **paired differences** `d_i` over the **same Rimsky A/B sycophancy
  pairs CAA uses** (maximizes CAA comparability — identical stimuli, identical read site). Zou
  et al.'s LAT can alternatively PCA the **stacked raw reps** (syc and non-syc reps as separate
  labeled points) or use a bespoke RepE-style stimulus template. I chose paired-diff-PCA because
  it holds the stimulus set identical to CAA (the whole point of the contrast). **Lead: confirm
  paired-diff PCA on CAA's pairs** (recommended), or request stacked-reps / a RepE stimulus
  template (would be a second, less-controlled extraction variant).

- **D-7 — n_components.** LAT's reading vector is the **top-1** PC. **Default:** `--n-components
  1`. Exposed for a sensitivity check (a top-k subspace steer would be a different operator and
  would break the single-vector CAA parity). **Lead: confirm top-1** (recommended, matches LAT).

- **D-8 — Reproduction-gate threshold & sweep grid.** CAA used **+25 pts** clean sycophancy gain
  with `--sweep-layers 12,14,16,18,20 --coeffs 4,8,12`. **Default:** the same threshold and grid
  (RepE is the same behavior/model/injection as CAA, so CAA's grid is the right prior; the
  reading vector is extracted per-layer from the same reps). **Lead: confirm +25 and the grid,**
  or adjust if the PCA direction needs a different coefficient range than the mean-diff (the
  sweep will reveal this; the gate picks the largest non-degenerate gain ≥ threshold, else an
  honest null, exactly as CAA).

---

## 8. Interfaces (both files `python3 -m py_compile`-clean — verified)

`exp/repe_steer.py` (mirrors `caa_steer.py`):
- `build_chat`, `is_sycophantic`, `sycophancy_rate`, `sycophancy_token_ids`, `wu_span_basis`,
  `cos_dir_wu_span` — **re-exported from `caa_steer` verbatim** (identical chat template,
  classifier, W_U geometry). Model load via `battery.load_model` (Qwen2.5-7B-Instruct, identical
  to CAA). `actlib` imported the SAME way as CAA (adds `<repo>/tools` to `sys.path`; the launch
  tarball needs `exp/` + `tools/actlib/`).
- `read_vectors_pca(...)` — **the one new component**: LAT top-PCA reading vector at the CAA read
  site/token, sign-aligned, unit-normalized; returns the SAME dict shape as `C.caa_vector`
  (`layer`, `v_hat`, `raw_norm`, `n_pairs`, + PCA diagnostics) so the driver drops it straight
  into the CAA stage1/stage2 flow. Reuses `C._last_token_resid_post` so the activation read is
  byte-identical to CAA.
- `RepEMethod` — `generate(prompt, coeff, mode)` for base/native/first with the resid_post
  additive hook + KV-baked E_first (copied from `CAAMethod` so κ is comparable in `single`
  mode); `generate_with_fixed_vector(...)` for the floor / W_U-secondary controls;
  `inject_layers` ∈ {single, all}.
- Battery helpers matching CAA: `position1_logit_delta`, `first_token_flip_count`,
  `teacher_forced_stepkl_native`, `kv_baked_first_sanity` (each dispatches over the active
  layer(s); one-layer = CAA-identical).

`exp/run_repe.py` (mirrors `run_caa.py`):
- argparse flags matching CAA + the new ones: `--device --dtype --n --n-calib --n-extract
  --tokens --n-boot --sweep-layers --coeffs --inject-layers --n-components --no-mean-center
  --repro-threshold --stage --smoke --outdir`.
- Same Rimsky sycophancy dataset loader + Anthropic fallback + held-out split as `run_caa`.
- stage1: layer×coeff sweep + ≥ +25 pts reproduction gate (disk-staged `stage.json`, resumable;
  reading vectors cached per layer as `repe_vec_L{L}.pt` so `all` mode can add them together);
  stage2: battery reusing `exp/battery.py` (TF-KL headline control = ρ, random-matched-norm floor
  ×3, verdict from CI bounds + cell_valid, Amendment-2 dose-response if the control voids); writes
  `results_full.json` + `report.md` incrementally (courier-safe, per the CAA ops lesson).
- AWS/staging conventions honored: `--outdir` default under
  `runs/steering-content-audit/2026-07-08-repe-7b`, `--smoke`, `--stage all`, resumable
  `stage.json`. DLAMI python `/opt/pytorch/bin/python`; launch tarball must include
  `exp/` + `tools/actlib/` (a prior run died on omitting actlib). **This note and the code do
  NOT launch anything.**

---

## 9. What is REUSED vs NEW (diff from CAA)

| Component | Source | Status |
|---|---|---|
| Read site/token (resid_post @ answer-letter token) | `caa_steer._last_token_resid_post` | **reused (identical)** |
| **Direction extraction** | `repe_steer.read_vectors_pca` | **NEW (PCA/LAT top-comp, sign-aligned)** |
| Additive resid_post hook + base/native/first loop | `caa_steer.CAAMethod._gen` pattern | reused (single-layer identical) |
| `--inject-layers single\|all` | `repe_steer.RepEMethod` | **NEW (flag; multi-layer path)** |
| KV-baked E_first + `kv_baked_first_sanity` | mirror of CAA | reused |
| `position1_logit_delta` / `first_token_flip_count` / `teacher_forced_stepkl_native` | mirror of CAA | reused (dispatch over layers) |
| Sycophancy classifier + eval prompts + dataset | `caa_steer` / `run_caa` | **reused verbatim** |
| TF-KL control, floor, ρ/κ CI, degeneracy gate, verdict, dose-response | `battery.py` / `run_caa` | reused verbatim |

**Net new surface: `read_vectors_pca` (LAT extraction) + the `inject_layers` flag/path.** The
CAA-vs-RepE contrast differs ONLY in mean-diff-vs-PCA (single-layer) — the design goal.

---

## 10. Correctness risks not resolvable without a GPU (flagged for launch)

- **R1 — PCA sign alignment (highest).** The sign is set by `mean_i(d_i · v̂)`; this is correct
  iff the difference convention (`syc − non_syc`) is preserved end-to-end (it is: same
  `answer_matching` letter logic as CAA). Cannot be behaviorally verified without the model.
  **Launch check:** confirm `label_frac_aligned` ≳ 0.5 and the +25 sweep shows a *positive*
  gain in `stage1_sweep.json`; a large *negative* gain would signal a flipped sign.
- **R2 — Read-site parity with CAA.** `repe_steer` calls `C._last_token_resid_post` directly, so
  the read is guaranteed byte-identical to CAA (same layer, same last/answer-letter token, same
  hook). Verified statically (shared function); no independent reimplementation to drift.
- **R3 — SVD orientation/convention.** `torch.linalg.svd` returns `Vh` with rows = right-singular
  vectors (principal components) — `Vh[0]` is the top PC. Standard and stable; the covariance-
  eigh fallback returns the same top direction. Sign is then fixed by R1's alignment.
- **R4 — Multi-layer κ interpretation.** In `--inject-layers all`, κ = E_first/E_native is
  computed but is **not** CAA-comparable (different injection pattern); the results/report label
  it so. Single-layer (default) is fully CAA-comparable.
- **R5 — bf16 numerical noise in PCA.** Reps are captured f32 (CAA's reader casts to f32) and PCA
  runs in f32, so the reading vector is computed in f32 even though the model is bf16 — matches
  CAA's precision for the direction. The steer is applied in the model dtype (bf16) exactly as
  CAA.
