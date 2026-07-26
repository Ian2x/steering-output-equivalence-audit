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
field. The submitted source and checker use the raw-artifact value `0.833333`.
No `rho`, gate outcome, or bounded conclusion changes.

## Interpretation of stored legacy labels

Some historical JSON files contain the development labels `Genuine` and
`Dissolved`. Those labels are retained to preserve artifact integrity; they
are not the paper's claim language. The paper reports output-reproducibility
measurements, coherent-window qualifications, and controller-fit limitations.
