# Known Issues

Defects and interpretive limits that a reader auditing these artifacts should
know about before reusing them. Each entry states what the issue is, where it
lives in the shipped bytes, what the manuscript does about it, and how to check
the claim independently. None of these are hidden in the paper; this file
collects them in one place so the supplement can be audited on its own.

## 1. The refusal dose-response ships under an NLL-exempt degeneracy gate

The preregistered generation-quality gate has three lines: repetition
`rep > 2*base_rep + 0.1`, length `median_len < 0.5*base_median_len`, and
likelihood `nll > 3*base_nll`. The refusal arm as implemented exempts the
likelihood line. The shipped grid rows therefore carry two flags
side by side: `gate.raw_tripped` is the strict preregistered verdict with the
likelihood line counted, and `gate.degenerate` is the NLL-exempt verdict that
the run actually used. The grid lives in
`results/2026-07-07-refusal-arm/dose_response_fine.json`.

The two conventions do not agree:

| Convention | Clean dose scales | Maximum effect ratio | Meets Amendment 2's three-clean-scale plateau criterion |
| --- | --- | --- | --- |
| Strict, likelihood line counted | 2 (`0.1`, `0.2`) | `0.15107913669064738` | No |
| NLL-exempt, as shipped | 5 (`0.1`–`0.5`) | `0.2589928057553957` | Yes |

The manuscript reports the strict maximum as primary and reports the arm as
gate-sensitive rather than as a clean plateau. The NLL exemption is recorded as
a post-hoc, outcome-affecting deviation.

Earlier releases of `reproduce_checks.py` computed the NLL-exempt maximum
`0.2589...` and returned it under the key name `strict_plateau_pass`. That key
name was a mislabel: it reported the exempt result while naming the strict one.
The key is gone. `check_refusal_plateau()` now re-derives all three gate lines
from the stored `eval_baseline_refs`, cross-validates every stored per-row flag
against recomputation, reports both conventions under
`gate_conventions.strict_preregistered_nll_counted` and
`gate_conventions.nll_exempt_as_shipped`, and asserts that the strict
convention does not clear Amendment 2 — if it ever did, the manuscript's
gate-sensitive reading would be wrong and the checker fails loudly.
`DERIVED_CHECKS.json` is the current checker's output.

## 2. `kappa` is not measured on a single intervention schedule

`kappa = E_first / E_native` is a per-cell cascade diagnostic: the intervention
is applied only in an early window and then withdrawn. Two different early
windows are used across the arms, and this was discovered after the runs were
frozen.

**Prefill-only** — the write is withdrawn before the first generated token is
processed, so it shapes only that token's own distribution and then persists
solely through the cached keys and values it already wrote:

- `code/battery.py`, `positions="last_prompt"` via `actlib` — synthetic anchor,
  activation addition, function vector, task vector.
- `code/refusal_direction.py`, `generate_chat(mode="first")` — the hook gate is
  cleared at `step == 0`, after the prefill forward has produced the first
  generated token.

**Prefill+1** — the write remains active through the forward pass that
*processes* the first generated token, so it also shapes the second token's
distribution:

- `code/caa_steer.py`, `CAASteerMethod._gen(apply="first")`.
- `code/sae_steer.py`, the corresponding manual loop (see its comment at the
  `step == 0` branch).

The CAA and SAE `kappa` values are therefore measured under a strictly larger
intervention window than the others. The direction of the resulting bias is not
determined a priori, so `kappa` should be compared within a schedule and not
across schedules. The manuscript tags every reported `kappa` with its schedule
and does not rank cells by `kappa` across schedules.

No `rho` value, gate, or verdict depends on this. Every cell whose `rho` is
reported uses `E_native` in the denominator; the only cells that define `rho`
against `E_first` are the task-vector cells, where a single KV-baked write is
the native intervention and the schedule is prefill-only by construction.
Standardizing the window would require rerunning the CAA and SAE cells, which
has not been done.

## 3. The full-vocabulary fidelity diagnostics are off-policy and are not bounds

`coverage` and `changed_top1_recovery` in `heldout_footprint_fidelity` are
measured on teacher-forced, base-generated prefixes, while the behavioral
evaluation that produces `rho` is closed-loop on the controller's own visited
prefixes. Prefix distribution shift can make an off-policy fidelity estimate
either optimistic or pessimistic; it does not bound the on-policy value in
either direction. The manuscript reads these numbers as optimistic diagnostics,
because the distiller's fit and the diagnostic share a prefix distribution that
the behavioral evaluation does not, but that is an expectation about this
particular fit rather than a guarantee.

## 4. Per-prompt structure is not retained in the shipped fidelity payloads

`heldout_footprint_fidelity` in
`results/2026-07-10-output-footprint-distill/fv.json` and `taskvec.json` is
aggregate only: 205 of 398 changed-top-1 positions recovered for the function
vector (`0.5150753768844221`) and 59 of 174 for the task vector
(`0.3390804597701149`). Those positions are clustered within prompts, and the
artifacts do not retain the grouping that a cluster-robust interval would
require. The manuscript therefore reports no interval on these rates, and the
function vector's margin over the preregistered `0.5` fidelity floor — six
positions out of 398 — should not be read as a resolved separation between the
two fits. Both are reported as fidelity-limited.

## 5. `run_refusal.py` ships a verdict-logic defect and corrupt point fields

Two defects in `code/run_refusal.py`, both disclosed in the manuscript and both
re-derived by `reproduce_checks.py` under `derived.refusal_cell_defects`, so
neither has to be taken on trust.

**Verdict logic.** The "Dissolved" branch reads
`rho_lo >= 0.9 and native_clean and E_over_floor >= 3.0` and omits
`control_clean`. The shipped `results/2026-07-07-refusal-arm/results_full.json`
therefore records `verdict.class = "Dissolved"` alongside
`verdict.cell_valid = false` and `verdict.gate_clean_control = false`: the
control had reached its refusal drop by degenerating into repetition loops
(three-gram repetition `0.968` against a `0.105` line). Amendment 2 forbids a
degenerate control from certifying reproduction, so **the corrected verdict for
that cell is Mixed, not Dissolved.** The manuscript never uses this cell; the
refusal row it reports comes from the Amendment-2 dose ladder instead.

**Point fields.** `battery.bootstrap_ratio_ci` computes its point estimate as
`(num.mean() - base.mean()) / max(den.mean() - base.mean(), 1e-9)`. The refusal
effect is a *suppression*, so the denominator is negative, the clamp fires, and
the stored `rho.point` and `kappa.point` are artifacts of order `-1e9`. The
bootstrap replicates recompute the denominator without the clamp, so the
intervals are correct, and the preregistered decision rules read intervals. The
manuscript recomputes both ratios from the recorded rates —
`kappa = (0.94 - 0.0267) / (0.94 - 0.0133) = 0.986` — and says so in the Table 3
notes. Any other arm whose native effect is a suppression would hit the same
clamp; among the shipped arms, only refusal does.

## 6. A dormant `--smoke` fall-through remains in `code/run_refusal.py`

`code/run_refusal.py:878` still reads
`if not stage1["reproduced"] and not args.smoke:`. In the cloned drivers this
pattern let a smoke run fall through to the stage-2 battery with
`chosen = None`, which crashes. It is dormant here because the refusal
direction reproduces even at smoke sample sizes, so the branch is never taken;
the arm's banked results are unaffected and its code was deliberately left
untouched. `code/run_caa.py` and `code/run_repe.py` in this supplement carry the
repaired form (`if not stage1["reproduced"]:` with a smoke-aware forced pick).
Repair `run_refusal.py` before reusing it.

## 7. Some reported quantities cannot be recomputed from what ships here

- **Per-prompt effect vectors ship for four of the eight audited cells.** The
  rest ship aggregates plus the code paths that produced them, so the bundled
  checker is a transcription-consistency check over aggregates and does not
  recompute bootstrap intervals from per-prompt data. The manuscript's
  reproducibility statement says this explicitly.
- **The layer-14 CAA sycophancy battery JSON is not in this supplement.** The
  manuscript reports that cell (`rho = 0.417 [0.305, 0.526]`,
  `kappa = 0.926`, prefill+1) as a layer-sensitivity check on the layer-18 row.
  Its numbers cannot be re-derived from these bytes.
- **Cluster structure is not retained** in the held-out fidelity payloads; see
  issue 4.

## 8. Cross-hardware and precision caveats on the exploratory branches

These affect the reproducibility branches reported in the manuscript's appendix,
not the counted cells.

- **RePS matched-control run stopped at an integrity guard.** The persisted
  steering vector loaded byte-exactly on the control host, but the preregistered
  eight-of-eight greedy continuation parity check returned seven of eight on
  replacement GPU hardware (an L4 rather than the original A10G), so no control
  generations exist. The RePS *reproduction* stands; what failed is
  cross-hardware generation parity.
- **The function-vector cached-versus-full-recompute text parity is 7/8** and is
  reported as a disclosed backend diagnostic. Task-vector parity is 8/8.
- **The Pythia distiller arms ran in bfloat16.** Positions at the low end of the
  absolute-KL axis (0.03 nats) carry reduced-precision numerics. The behavioral
  rates and gate statistics the verdicts rest on are not sub-nat quantities.

## 9. `code/draw_rho_kappa_pdf.py` is deprecated and refuses to run

It emitted a PDF referencing non-embedded Type 1 Helvetica, which renders with
invisible text in some viewers, and it duplicated the summary figure's data as
hard-coded constants that then drifted from the manuscript. The shipped figure
is produced by `code/plot_paper_summary.py` (Matplotlib, TrueType fonts
embedded). The deprecated script is kept only so the historical record is
complete; it exits immediately rather than let the two generators diverge again.

## Reporting

These artifacts are frozen. If you find a further discrepancy between the
shipped bytes and a manuscript claim, it is a defect in the manuscript, not a
correction to be made silently in the artifacts.
