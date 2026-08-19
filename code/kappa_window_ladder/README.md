# Window-standardization and calibration-ladder drivers

Verbatim copies of the drivers that produced
`results/2026-08-06-kappa-window-standardization/` (Post-hoc disclosure 21;
Sections 3.5 and 5.6) and `results/2026-08-06-e3-calibration-ladder/`
(Appendix I). They are kept in this subdirectory, rather than beside the
original drivers in `code/`, for one reason: `replay_caa_semantic_check.py`
here is the fail-closed CAA window replay (run id
`20260806-kappa-window-standardization`) and shares its file name with the
earlier Amendment-20 semantic-check replay in `code/`, which produced
`results/2026-07-10-caa-semantic-check/replay_generations.json`. Both are
retained under their original names.

| File | Role |
|---|---|
| `run_kappa_window_sae.py` | Regenerates the four historical SAE conditions behind an exact-replay gate, then measures the prefill-only `E_first` condition (`sae_window_raw.json`). |
| `replay_caa_semantic_check.py` | Matched-hardware CAA replay for both behaviors under both windows (`caa_window_raw.json`); the new condition does not run unless the historical hit counts reproduce exactly. |
| `test_kappa_window.py` | Unit tests for the two window drivers (hook-window activity, gate predicates). |
| `run_e3_calibration_curve.py` | Calibration-size ladder for the full-vocabulary distiller (400 to 6,400 rows), including the 400-row parity gate against the shipped fits. |
| `reduce_e3_stage_a.py`, `reduce_e3_calibration_curve.py` | Independent reductions of the ladder artifacts (`independent_reduction.json`, parity decisions). |
| `test_e3_calibration_curve.py` | Unit tests for the ladder driver. |

Running notes. These drivers import `battery`, `run_sae`, `sae_steer`,
`caa_steer`, `run_caa`, and `run_output_footprint_distill` from the directory
they are executed from, so copy them beside the files in `code/` (or put
`code/` on `PYTHONPATH`) before a rerun; when copying, rename this
directory's `replay_caa_semantic_check.py` so it does not overwrite the
Amendment-20 file. Like the other drivers, they resolve run and project paths
relative to a project checkout and import the shared activation-capture
library shipped in `code/actlib/`. The two reduction scripts take
their input directories as arguments (`--rundir`/`--outdir`, `--shipped-dir`)
and need only the ladder JSON artifacts plus NumPy; the tests need torch,
transformers, and pytest.
