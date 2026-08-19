# Result Provenance

All paths are relative to this supplement. The paper's displayed values are
rounded from the listed JSON fields; `reproduce_checks.py` verifies the
unrounded values.

| Paper result | Source artifact | Load-bearing fields |
|---|---|---|
| Synthetic positive anchor | `results/2026-07-06-a0-anchor-r2/results_full.json` | `rates`, `rho`, `kappa`, `degeneracy_gates` |
| Function vector | `results/2026-07-06-a1-anchor/results_full.json` | `rates`, `rho`, `kappa`; void static control retained |
| Activation Addition | `results/2026-07-06-actadd-arm/results_full.json` | `rates.E_native`, `rates.control`, `rho`, gates |
| Refusal ablation | `results/2026-07-07-refusal-arm/dose_response_fine.json` | `grid`, `adjudication`; strict coherent-window result |
| In-context task vector | `results/2026-07-08-taskvec-7b/results_full.json` and `steelman_dose_f*.json` | native effect, static anchor, corrected absolute-KL grid |
| SAE feature steering | `results/2026-07-07-sae-arm/results_full.json` | `rates`, `rho`, `kappa`, gates |
| CAA sycophancy | `results/2026-07-10-caa-semantic-check/source/sycophancy/results_full.json` | `rates`, `rho`, `kappa`, gates |
| CAA corrigibility-match | `results/2026-07-10-caa-semantic-check/source/corrigibility/results_full.json` | `rates`, `rho`, `kappa`, gates |
| Full-vocabulary controller frontiers | `results/2026-07-10-output-footprint-distill/{synthetic,fv,taskvec}.json` | `frontier`, `decision`, `heldout_footprint_fidelity`, `oracle_guard` |
| CAA semantic robustness | `results/2026-07-10-caa-semantic-check/semantic_judged.json` | cleaned alignment, paired differences, surface-only rates, verdicts |
| Calibration-size ladder (Appendix I) | `results/2026-08-06-e3-calibration-ladder/{fv,taskvec}_curve.json` | `sizes[].fit.effective_rank`, `sizes[].fidelity.derived`, `sizes[].fidelity.prompt_cluster_bootstrap_95`, `frontier_at_largest` |
| Calibration-ladder parity gate | `results/2026-08-06-e3-calibration-ladder/{fv,taskvec}_parity_decision.json` | 47 per-arm checks against the shipped 400-row fits, 40 of them exact equality |
| Calibration-ladder outcome | `results/2026-08-06-e3-calibration-ladder/independent_reduction.json` | paired recovery deltas, plateau and material-gain predicates, `outcome_map.row` |
| Standardized `kappa`, SAE cell (Sections 3.5, 5.6) | `results/2026-08-06-kappa-window-standardization/sae_window_raw.json` | `conditions.*.hits` per-prompt vectors for both windows, `gate0`, `gate1`, `kappa`, `rho_confirmation` |
| Standardized `kappa`, both CAA cells | `results/2026-08-06-kappa-window-standardization/caa_window_raw.json` | `behaviors.*.conditions.*.hits`, `behaviors.*.gate0` exact-replay predicates, `behaviors.*.metrics.kappa`, `behaviors.*.metrics.precision`, `decision` |
| Window-standardization environment | `results/2026-08-06-kappa-window-standardization/environment.json` | torch/transformers/accelerate versions and CUDA device for the rerun |
| Window-standardization drivers | `code/kappa_window_ladder/run_kappa_window_sae.py`, `code/kappa_window_ladder/replay_caa_semantic_check.py` | exact-replay gate, both windows; the CAA file shares its name with the Amendment-20 semantic replay in `code/` (see the subdirectory README) |
| Calibration-ladder drivers | `code/kappa_window_ladder/run_e3_calibration_curve.py`, `reduce_e3_stage_a.py`, `reduce_e3_calibration_curve.py` | ladder fits, 400-row parity gate, independent reduction |
| Gate triplets and gate-sensitivity sweep (Section 3.4, Appendix H) | `degeneracy_gates` / `gates` blocks of the per-cell `results_full.json` files above | 3-gram repetition, median length, mean NLL per condition; the CAA driver's `cell_valid` counts only the first two lines (`KNOWN_ISSUES.md`, issue 8) |

## Intervention-window standardization

The three cells whose shipped drivers measured `E_first` on the wider
prefill+1 window — SAE feature steering and both CAA cells — were rerun under
both windows after the release candidate was built (Post-hoc disclosure 21).
The rerun is gated on exact reproduction of each cell's historical baseline,
native, control, and prefill+1 hit counts, so the standardized prefill-only
estimates extend the shipped artifacts rather than replacing them. The
originally shipped `results_full.json` files are unchanged and remain the
source for every `rho` value and gate; `reproduce_checks.py` re-derives both
windows from the per-prompt hit vectors above and asserts that the replayed
`rho` equals the frozen one in all three cells.

One release-time edit applies to `sae_window_raw.json`. That cell ran locally,
so its `source.banked_result` and `source.prompts` fields were written as
absolute filesystem paths. Both have been rewritten as project-relative paths;
the accompanying `banked_result_sha256` and `prompts_sha256` fields are
untouched, so the inputs remain identified by content. No derived quantity
reads these fields, and `DERIVED_CHECKS.json` is byte-identical before and
after the rewrite.

## Corrections retained in the record

The final paper uses the repaired synthetic anchor
`2026-07-06-a0-anchor-r2`; the earlier
`2026-07-05-a0-anchor` artifact is included so the superseded run remains
inspectable.

During the final source audit, the CAA-corrigibility cascade-share point was
re-derived from the shipped rates:

```text
baseline = 0.27
native   = 0.45
first    = 0.42
kappa    = (0.42 - 0.27) / (0.45 - 0.27) = 0.833333...
```

The bootstrap interval is `[0.538461..., 1.066667...]`. An earlier tracker
and figure had accidentally copied the interval's lower bound into the point
field. The checker uses the raw-artifact value `0.833333`. That value is the
prefill+1 measurement; the paper's Table 3 reports this cell's standardized
prefill-only `kappa`, `0.722 [0.400, 1.000]`, from `caa_window_raw.json`
above and retains `0.833 [0.538, 1.067]` as the window-sensitivity row. No
`rho`, gate outcome, or bounded conclusion changes.

## Interpretation of stored legacy labels

Some historical JSON files contain the development labels `Genuine` and
`Dissolved`. Those labels are retained to preserve artifact integrity; they
are not the paper's claim language. The paper reports output-reproducibility
measurements, coherent-window qualifications, and controller-fit limitations.
