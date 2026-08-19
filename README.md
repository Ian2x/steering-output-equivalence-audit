# Auditing Activation Steering with Matched-Budget Output Controls

Code and result artifacts for the output-equivalence audit described in the paper
*Auditing Activation Steering with Matched-Budget Output Controls* (Ian Wang, 2026).

The audit asks a single question about each steering method: after the method has
reproduced a real held-out behavioral effect, how much of that effect does an
externally constructed output-interface controller reproduce at a matched
interface budget? The reproduced fraction is `rho = E_control / E_native`, with a
companion temporal quantity `kappa = E_first / E_native` measuring how much of the
native effect is established through the first generated token alone.

Model weights, activation caches, and other large intermediate tensors are not
included. Everything needed to re-derive the paper's numbers from stored results
is.

## Contents

| Path | What it is |
|---|---|
| `results/` | Shipped JSON artifacts for every countable paper row, the full-vocabulary controller study, the CAA semantic robustness check, the intervention-window standardization rerun (`2026-08-06-kappa-window-standardization`), and the calibration-size ladder (`2026-08-06-e3-calibration-ladder`) |
| `code/` | Experiment drivers, controller and gate implementations, scoring, and plotting |
| `code/actlib/` | The shared activation-capture library the drivers import (model loading, hook registration, KV-baked one-shot patches) |
| `code/kappa_window_ladder/` | The window-standardization and calibration-ladder drivers, kept apart because one file name clashes with an earlier driver in `code/` (see its `README.md`) |
| `reproduce_checks.py` | Standard-library checker that re-derives headline quantities from the shipped JSON bytes |
| `DERIVED_CHECKS.json` | The checker's own output, stored so you can diff a fresh run against the release |
| `RESULT_PROVENANCE.md` | Mapping from paper claims to source artifacts |
| `MANIFEST.sha256` | Checksums for every file in this repository |
| `KNOWN_ISSUES.md` | Honest record of driver defects, accounting gaps, and their scope |
| `requirements.txt`, `requirements.lock.txt` | Environment records for full model-backed reruns |

## Artifact-only verification

The numerical audit requires only Python 3 and downloads no models:

```bash
python3 reproduce_checks.py
```

The checker recomputes `rho` and `kappa` from stored rates wherever those ratios
are defined, verifies all countable rows, checks both full-vocabulary controller
frontiers, reports the semantic-judge surface-only rates, and adjudicates the
refusal dose response under **both** gate conventions — the preregistered strict
gate (maximum reproduction 0.151, two clean scales) and the implemented
NLL-exempt gate (plateau 0.259, five clean scales). The paper reports the strict
reading as primary and the NLL-exempt reading as a disclosed post-hoc secondary;
this checker lets you confirm both numbers yourself rather than taking the
disclosure on trust. It also re-derives the two refusal-driver defects recorded
in `KNOWN_ISSUES.md` — the clamped `rho` and `kappa` point fields and the
verdict block that contradicts them — so those are recomputable rather than
merely asserted.

The checker also re-derives the intervention-window standardization behind
Post-hoc disclosure 21. Three cells — SAE feature steering and both CAA cells —
originally measured `E_first` over a wider prefill+1 window than the rest of the
paper. Each was rerun under both windows, gated on exact reproduction of its
historical baseline, native, control, and prefill+1 counts, so the standardized
prefill-only estimates extend the shipped artifacts rather than replacing them.
From the per-prompt hit vectors in
`results/2026-08-06-kappa-window-standardization/`, the checker recomputes both
windows for all three cells, asserts that the replayed `rho` equals the frozen
value in each, and reproduces the paired discordant-pair counts and the pooled
post-hoc exact test the paper reports as a secondary analysis.

To verify file integrity:

```bash
shasum -a 256 -c MANIFEST.sha256
```

## Regenerating the figures

All three paper figures regenerate from two scripts, without model inference:

```bash
python3 code/plot_paper_summary.py
python3 code/plot_output_frontiers.py --out reproduced/rho_vs_absolute_kl.png
```

`plot_paper_summary.py` writes the rho/kappa map (Figure 1) and the CAA semantic
robustness figure (Figure 3) into `paper_figures/`; `plot_output_frontiers.py`
writes the absolute-KL frontiers (Figure 2) to the `--out` path, reading the
6,400-row refit series from `results/2026-08-06-e3-calibration-ladder/`. Each script
emits a PNG and a PDF; the PDFs are the ones the paper embeds, and they
reproduce the published files byte for byte apart from the embedded creation
timestamp. `code/draw_rho_kappa_pdf.py` is retained only for the historical
record and refuses to run: it emitted non-embedded Type 1 fonts, which rendered
as invisible text in several PDF viewers.

## Full reruns

The experiment drivers under `code/` require GPU hardware, model downloads, and
in two cases an OpenAI API key for the automated semantic judge (read from the
`OPENAI_API_KEY` environment variable; no credentials are stored in this
repository). The drivers import a small shared activation-capture library,
`actlib`, which ships in `code/actlib/`; they resolve it from their own
directory, so no path configuration is needed. That copy is the library the runs
used; only comments and docstrings that named unrelated internal projects were
removed, no code line changed. The artifact-only verification and figure
regeneration above do not import it. Reruns will not reproduce the shipped bytes
exactly across different hardware — see `KNOWN_ISSUES.md` for a documented
cross-backend generation-parity failure that stopped one arm before it produced
an artifact.

## Scope and claim boundary

This audit measures behavioral output equivalence under named controller families
and coherent absolute-KL budgets. It does not identify what a steering vector
represents, estimate a causal mediation effect, or establish that a low-`rho`
behavior is impossible to reproduce with a richer output policy. The audited
cells are confounded by task, model, metric, and gate strictness, so the results
describe cells rather than methods.

## License

Code in `code/` and `reproduce_checks.py` is released under the MIT License (see
`LICENSE`). The result artifacts under `results/` and the accompanying
documentation are released under CC BY 4.0, matching the paper's license.

## Citation

A citation entry will be added here once the preprint has an arXiv identifier.
