# Known issues

This file records defects and accounting gaps found in the audit's own tooling.
None of them changes a reported number, but each one is the kind of thing a
reader should be able to check rather than take on trust.

## `code/run_refusal.py` — verdict logic and point estimates

Two defects were found in this driver during adjudication of the refusal arm:

1. **Verdict logic.** It printed `VERDICT: Dissolved` from the `rho` confidence
   interval while its own `cell_valid` flag was `False`. The primary control had
   reached its refusal-rate drop by degenerating into repetition loops
   (repetition 0.968), and the preregistered rules are explicit that a degenerate
   control can never count as reproducing a behavior. The corrected verdict was
   Mixed, pending a finer dose response, and that is what the paper reports.
2. **Point-estimate arithmetic.** The `rho` and `kappa` point estimates printed by
   this driver used a faulty division. The confidence intervals were computed
   correctly, and the preregistered decision rules read the interval bounds, so no
   adjudication depended on the bad points. The paper's refusal `kappa` (0.986) is
   recomputed from the recorded base, native, and first-token rates.

A third, dormant defect is shared with the other cloned drivers: a
`not stage1["reproduced"] and not args.smoke` guard lets a smoke run fall through
to the full battery on an unselected configuration. It never fired for the refusal
or CAA arms, whose directions reproduce even at smoke sample sizes, and their
proven code was deliberately left untouched. **Fix all three defects before
reusing or extending this driver.**

## `reproduce_checks.py` — resolved key-name mislabel

Earlier releases of the checker reported the refusal plateau under the key
`strict_plateau_pass` while actually computing it from the implemented,
NLL-exempt gate. The checker in this repository computes and reports both
conventions under unambiguous key names, and independently re-derives the strict
maximum of 0.151 that the paper reports as primary.

## Accounting gaps

- The layer-14 CAA sycophancy battery JSON is not included. That cell is reported
  in the paper's appendix as a layer-sensitivity check (`rho` 0.417, `kappa` 0.926,
  Mixed); its aggregate values are recorded in the paper but its raw battery record
  is not part of this artifact set.
- Per-prompt effect vectors ship for four of the eight audited cells; the rest
  ship aggregates plus the code that produced them.
- The antonym and refusal datasets resolve from their upstream public sources at
  run time rather than being redistributed here.

## Cross-backend generation parity

One breadth arm (RePS) reproduced its steering vector byte-exactly but failed a
frozen eight-continuation exact-regeneration check at 7/8 on replacement GPU
hardware (an L4 in place of the original A10G). Under the preregistered stop rule
no control artifact was produced, so this repository contains that arm's code but
no results. Treat exact-byte reproduction across different accelerators as
something to verify, not assume.

## Precision

The full-vocabulary distiller arms ran in bfloat16. The behavioral rates and gate
statistics they feed are counts and rates rather than sub-nat quantities, but the
low end of the absolute-KL axis (0.03 nats) carries reduced-precision numerics.
