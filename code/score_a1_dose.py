#!/usr/bin/env python3
"""Cross-frac aggregator for the steelman budget dose-response.

Reads the per-frac steelman_dose_f*.json files produced by the validated
run_a1_steelman.py / run_a1_steelman_taskvec.py (--budget-kl f*B*) and applies
the PRE-REGISTERED Amendment-6 / Amendment-7 decision rule (plan.md §11)
verbatim:

  SURVIVES  <=> at EVERY clean (non-degenerate) budget-frac, every bounded
                input-conditional rung (k <= 16) has rho_hi <= 0.3, AND there
                are >= 3 clean sub-degenerate budget-fracs.
  ARTIFACT  <=> at some clean budget-frac a bounded rung (k <= 16) reaches
                rho_lo >= 0.9.
  k=full never calls Dissolved (labeled upper anchor).

Gate semantics are FROZEN project-wide: a gate-tripped (degenerate) cell is
VOID -- it yields NO verdict (cannot certify OR refute). So the decision rule
is evaluated over NON-DEGENERATE bounded cells only. We ALSO print the literal
"all bounded rungs incl. degenerate" reading and FLAG any divergence.

Validity readouts (not gates, but reported):
  (i)  k=0 dose-response == the static null at each budget (determinism check);
       must equal the frozen static-null rho at frac 1.00.
  (ii) input-conditional LIFT  rho_k<=16(f) - rho_0(f)  at each clean frac --
       the quantity that answers HOLE 1 directly. Lift ~ 0 across clean
       budgets is the strongest SURVIVES evidence.

GENERALIZATION (Amendment 7): the run directory is selectable via --rundir
(default = the FV a1-anchor dir, so the pre-existing FV usage is preserved
byte-for-byte). B* and the frozen static-null rho anchor are read FROM THE RUN's
own dose JSONs (meta.B_star_frozen and sanity.static_null_rho) rather than
hardcoded FV constants, so the same aggregator scores the task-vec run. The
FV constants remain as a documented cross-check when --rundir is the FV dir.

Pure stdlib file I/O -- no torch, no MPS. Safe to run while a sweep is live.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path


def default_fv_rundir():
    here = Path(__file__).resolve()
    supplement = here.parent.parent / "results" / "2026-07-06-a1-anchor"
    repo_run = here.parents[3] / "runs" / "steering-content-audit" / "2026-07-06-a1-anchor"
    return str(supplement if supplement.exists() else repo_run)


FV_RUNDIR = default_fv_rundir()
# Frozen FV anchors (Amendment 2 / run-1) — used ONLY as a cross-check when the
# aggregated run is the FV dir; the operative anchors are read from the JSONs.
FV_B_STAR = 0.7035311031341552
FV_STATIC_NULL_RHO_POINT = -0.11864406779661019
FV_STATIC_NULL_RHO_LO = -0.23076923076923075
FV_STATIC_NULL_RHO_HI = -0.0392156862745098

BOUNDED_KS = {"0", "1", "4", "16"}     # k <= 16
SURVIVE_HI = 0.3
ARTIFACT_LO = 0.9
MIN_CLEAN_FRACS = 3


def load_doses(rundir):
    """Return list of (frac, path, obj) sorted by frac ascending."""
    out = []
    for path in glob.glob(os.path.join(rundir, "steelman_dose_f*.json")):
        base = os.path.basename(path)
        # steelman_dose_f0.10.json  /  steelman_dose_f1.00.json
        frac_str = base[len("steelman_dose_f"):-len(".json")]
        try:
            frac = float(frac_str)
        except ValueError:
            continue
        with open(path) as fh:
            out.append((frac, path, json.load(fh)))
    out.sort(key=lambda t: t[0])
    return out


def read_anchors(doses):
    """Read (B_star, static_null_rho dict) from the run's own dose JSONs.

    B* = meta.B_star_frozen (consistent across a run's fracs). Static-null rho =
    sanity.static_null_rho (the frozen anchor the harness itself loaded for its
    k=0 determinism check). We take them from the frac-1.00 file if present (the
    canonical determinism point), else the first available file. Robust to either
    the FV or the task-vec run.
    """
    if not doses:
        return None, None
    # Prefer frac 1.00 for the anchor read (the determinism reference point).
    pick = None
    for frac, path, obj in doses:
        if abs(frac - 1.00) < 1e-6:
            pick = obj
            break
    if pick is None:
        pick = doses[0][2]
    m = pick.get("meta", {})
    b_star = m.get("B_star_frozen")
    sn = pick.get("sanity", {}).get("static_null_rho")
    # Fallback: some FV JSONs may carry the static null only in sanity; both do.
    return b_star, sn


def rung_by_k(obj, k):
    for r in obj["rungs"]:
        if str(r["k"]) == str(k):
            return r
    return None


def fmt_rho(r):
    rho = r["rho"]
    deg = "DEGEN" if r["gate"]["tripped"] else "clean"
    return (f"rho={rho['point']:+.3f} [{rho['ci_lo']:+.3f},{rho['ci_hi']:+.3f}] "
            f"acc={r['eval_acc']*100:4.1f}% rep={r['gate']['rep']:.3f} {deg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", default=FV_RUNDIR,
                    help="run dir with steelman_dose_f*.json (default = FV "
                         "a1-anchor dir; preserves the pre-existing FV usage).")
    args = ap.parse_args()
    rundir = args.rundir

    doses = load_doses(rundir)
    if not doses:
        print("no steelman_dose_f*.json files found in", rundir)
        return 2

    # Anchors read FROM THE RUN (generalization). B* and static-null rho.
    b_star, static_null = read_anchors(doses)
    if b_star is None or static_null is None:
        print("ERROR: could not read B*/static-null anchors from run JSONs "
              f"in {rundir} (meta.B_star_frozen / sanity.static_null_rho).")
        return 2
    SN_POINT = float(static_null["point"])
    SN_LO = float(static_null["ci_lo"])
    SN_HI = float(static_null["ci_hi"])
    is_fv_dir = os.path.abspath(rundir) == os.path.abspath(FV_RUNDIR)

    have_fracs = [f for f, _, _ in doses]
    expected = [0.04, 0.10, 0.13, 0.16, 0.19, 0.25, 0.40, 0.55, 0.70, 1.00]
    missing = [f for f in expected
               if not any(abs(f - g) < 1e-6 for g in have_fracs)]

    print("=" * 78)
    print("STEELMAN BUDGET DOSE-RESPONSE  (cross-frac aggregation, "
          "Amendment 6/7)")
    print("=" * 78)
    print(f"run dir: {rundir}")
    print(f"B* = {b_star:.6f}   fracs present: {have_fracs}")
    print(f"frozen static-null rho anchor (from run): {SN_POINT:+.6f} "
          f"[{SN_LO:+.6f}, {SN_HI:+.6f}]")
    if is_fv_dir:
        # Cross-check against the hardcoded FV constants (regression guard).
        ok_bstar = abs(b_star - FV_B_STAR) < 1e-9
        ok_sn = (abs(SN_POINT - FV_STATIC_NULL_RHO_POINT) < 1e-9
                 and abs(SN_LO - FV_STATIC_NULL_RHO_LO) < 1e-9
                 and abs(SN_HI - FV_STATIC_NULL_RHO_HI) < 1e-9)
        print(f"  [FV dir cross-check] B* matches FV const: {ok_bstar}; "
              f"static-null matches FV const: {ok_sn}")
    if missing:
        print(f"  ** INCOMPLETE: fracs still missing {missing} "
              f"(sweep in progress?) **")
    print()

    # -- per-frac / per-rung table ------------------------------------------
    print(f"{'frac':>5} {'kl':>7} {'k':>4}  {'detail':<58}")
    print("-" * 78)
    clean_fracs = []            # fracs with >=1 non-degenerate BOUNDED rung
    survive_ok_fracs = []       # clean fracs where every non-degen bounded rung has rho_hi<=0.3
    artifact_hits = []          # (frac,k) clean bounded cells with rho_lo>=0.9
    literal_survive_fail = []   # (frac,k) bounded cells (incl degen) with rho_hi>0.3
    k0_track_flags = []
    for frac, path, obj in doses:
        kl = obj["meta"].get("budget_target", float("nan"))
        nondegen_bounded = []
        for k in ["0", "1", "4", "16", "full"]:
            r = rung_by_k(obj, k)
            if r is None:
                continue
            tag = ""
            if k in BOUNDED_KS and not r["gate"]["tripped"]:
                nondegen_bounded.append((k, r))
            if k in BOUNDED_KS and r["rho"]["ci_hi"] > SURVIVE_HI:
                literal_survive_fail.append((frac, k, r["rho"]["ci_hi"]))
                tag = "  <== rho_hi>0.3 (literal)"
            print(f"{frac:>5.2f} {kl:>7.4f} {k:>4}  {fmt_rho(r):<58}{tag}")
        # clean-frac bookkeeping (VOID semantics: degenerate cells excluded)
        if nondegen_bounded:
            clean_fracs.append(frac)
            all_hi_ok = all(r["rho"]["ci_hi"] <= SURVIVE_HI
                            for _, r in nondegen_bounded)
            if all_hi_ok:
                survive_ok_fracs.append(frac)
            for k, r in nondegen_bounded:
                if r["rho"]["ci_lo"] >= ARTIFACT_LO:
                    artifact_hits.append((frac, k, r["rho"]["ci_lo"]))
        # validity (i): k=0 determinism vs frozen static null at frac 1.00
        r0 = rung_by_k(obj, "0")
        if r0 is not None and abs(frac - 1.00) < 1e-6:
            match = (abs(r0["rho"]["point"] - SN_POINT) < 1e-9
                     and abs(r0["rho"]["ci_lo"] - SN_LO) < 1e-9
                     and abs(r0["rho"]["ci_hi"] - SN_HI) < 1e-9)
            k0_track_flags.append((frac, match, r0["rho"]["point"]))
        print("-" * 78)

    # -- input-conditional lift  rho_k<=16(f) - rho_0(f) --------------------
    print()
    print("INPUT-CONDITIONAL LIFT   rho_k(f) - rho_0(f)   "
          "(bounded k; clean cells only)")
    print(f"{'frac':>5}  {'k=4':>10} {'k=16':>10}   "
          f"(blank = that cell degenerate/VOID)")
    print("-" * 78)
    for frac, path, obj in doses:
        r0 = rung_by_k(obj, "0")
        cells = []
        for k in ["4", "16"]:
            r = rung_by_k(obj, k)
            if (r is None or r["gate"]["tripped"] or r0 is None
                    or r0["gate"]["tripped"]):
                cells.append(f"{'':>10}")
            else:
                lift = r["rho"]["point"] - r0["rho"]["point"]
                cells.append(f"{lift:>+10.3f}")
        print(f"{frac:>5.2f}  {cells[0]} {cells[1]}")
    print("-" * 78)

    # -- decision -----------------------------------------------------------
    n_clean = len(clean_fracs)
    survives = (n_clean >= MIN_CLEAN_FRACS
                and len(survive_ok_fracs) == n_clean
                and n_clean > 0
                and not artifact_hits)
    artifact = len(artifact_hits) > 0

    print()
    print("=" * 78)
    print("DECISION (Amendment-6/7, VOID semantics on degenerate cells)")
    print("=" * 78)
    print(f"  clean budget-fracs (>=1 non-degen bounded rung): "
          f"{n_clean}  {clean_fracs}")
    print(f"  clean fracs with all non-degen bounded rho_hi<=0.3: "
          f"{len(survive_ok_fracs)}  {survive_ok_fracs}")
    print(f"  ARTIFACT hits (clean bounded rho_lo>=0.9): "
          f"{artifact_hits or 'none'}")
    print(f"  >=3 clean fracs required: "
          f"{'MET' if n_clean >= MIN_CLEAN_FRACS else 'NOT MET'}")
    print()
    if artifact:
        verdict = ("ARTIFACT (input-conditional null reproduces the method at a "
                   "clean budget)")
    elif survives:
        verdict = "SURVIVES (Genuine verdict withstands the steelman)"
    elif n_clean < MIN_CLEAN_FRACS:
        verdict = (f"INCONCLUSIVE (only {n_clean} clean fracs < "
                   f"{MIN_CLEAN_FRACS}; budget window too degenerate -- need "
                   f"lower/denser fracs)")
    else:
        verdict = ("INCONCLUSIVE (clean fracs present but SURVIVES bar not fully "
                   "met -- inspect)")
    print(f"  VERDICT: {verdict}")

    # -- transparency: literal vs VOID divergence + validity ---------------
    print()
    print("TRANSPARENCY / VALIDITY")
    if literal_survive_fail:
        print(f"  literal-reading rho_hi>0.3 cells (incl. degenerate/VOID): "
              f"{literal_survive_fail}")
        print("   -> divergence between literal and VOID readings; inspect "
              "before finalizing.")
    else:
        print("  no bounded cell (clean OR degenerate) has rho_hi>0.3 -- "
              "literal and VOID readings AGREE.")
    for frac, match, pt in k0_track_flags:
        print(f"  validity(i) k=0 @frac {frac:.2f}: static-null determinism "
              f"{'EXACT MATCH' if match else 'MISMATCH'} (point {pt:+.5f} vs "
              f"frozen {SN_POINT:+.5f})")
    if missing:
        print(f"  ** result is PARTIAL -- {len(missing)} frac(s) missing: "
              f"{missing} **")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
