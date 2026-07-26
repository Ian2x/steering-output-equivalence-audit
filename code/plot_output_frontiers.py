"""Plot absolute-KL output-equivalence frontiers from shipped JSON artifacts."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


REPO = Path(__file__).resolve().parents[3]
SUPPLEMENT_RESULTS = Path(__file__).resolve().parents[1] / "results"
if SUPPLEMENT_RESULTS.exists():
    OLD_RUNS = {
        "FV": SUPPLEMENT_RESULTS / "2026-07-06-a1-anchor",
        "Task vector": SUPPLEMENT_RESULTS / "2026-07-08-taskvec-7b",
    }
    NEW_RUN = SUPPLEMENT_RESULTS / "2026-07-10-output-footprint-distill"
else:
    OLD_RUNS = {
        "FV": REPO / "runs/steering-content-audit/2026-07-06-a1-anchor",
        "Task vector": REPO / "runs/steering-content-audit/2026-07-08-taskvec-7b",
    }
    NEW_RUN = (
        REPO / "runs/steering-content-audit/2026-07-10-output-footprint-distill"
    )
COLORS = {"0": "#7f8c8d", "4": "#2f6f8f", "16": "#2f8f5b", "full": "#8e44ad"}
# Deterministic per-rank x-offsets so coincident quantized rungs stay visible.
JITTER = {"0": -0.0022, "4": 0.0, "16": 0.0022, "full": 0.0044}
MATCHED_NOTES = {
    "FV": "matched point (KL 0.70): void",
    "Task vector": "matched diagnostic (KL 7.38): void",
}


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def old_rows(name: str) -> list[dict]:
    rows = []
    pattern = str(OLD_RUNS[name] / "steelman_dose_f*.json")
    for raw_path in glob.glob(pattern):
        data = load_json(Path(raw_path))
        target = float(data["meta"]["budget_target"])
        for rung in data["rungs"]:
            rank = str(rung["k"])
            if rank not in COLORS:
                continue
            rows.append({
                "target": target,
                "rank": rank,
                "x": float(rung["achieved_kl"]),
                "y": float(rung["rho"]["point"]),
                "lo": float(rung["rho"]["ci_lo"]),
                "hi": float(rung["rho"]["ci_hi"]),
                "void": bool(rung["gate"]["tripped"]),
            })
    return sorted(rows, key=lambda row: (row["rank"], row["x"]))


def new_rows(name: str) -> list[dict]:
    arm = "fv" if name == "FV" else "taskvec"
    data = load_json(NEW_RUN / f"{arm}.json")
    return [{
        "x": float(row["achieved_kl"]),
        "y": float(row["rho"]["point"]),
        "lo": float(row["rho"]["ci_lo"]),
        "hi": float(row["rho"]["ci_hi"]),
        "void": bool(row["gate"]["tripped"]),
    } for row in data["frontier"]]


def plot_panel(ax, name: str) -> None:
    old = old_rows(name)
    new = new_rows(name)

    by_target = {}
    for row in old:
        by_target.setdefault(row["target"], []).append(row)
    for target, rows in by_target.items():
        if target <= 0.35 and all(row["void"] for row in rows):
            ax.axvspan(target - 0.008, target + 0.008,
                       color="#d9d9d9", alpha=0.7, linewidth=0)

    for rank, color in COLORS.items():
        rows = [row for row in old if row["rank"] == rank and row["x"] <= 0.35]
        if not rows:
            continue
        clean = [row for row in rows if not row["void"]]
        if clean:
            ax.plot([row["x"] + JITTER[rank] for row in clean],
                    [row["y"] for row in clean],
                    color=color, alpha=0.55, linewidth=1.2, label=f"S=100 rank {rank}")
        for row in rows:
            # Void points keep their rank color; the "x" glyph marks voidness.
            marker = "x" if row["void"] else "o"
            ax.errorbar(row["x"] + JITTER[rank], row["y"],
                        yerr=[[row["y"] - row["lo"]],
                              [row["hi"] - row["y"]]],
                        fmt=marker, color=color,
                        markersize=4.5, capsize=1.5, alpha=0.8, linewidth=0.8)

    ax.plot([row["x"] for row in new], [row["y"] for row in new],
            color="#111111", linewidth=2.2, marker="s", markersize=4.5,
            label="Full-vocab per-step")
    for row in new:
        ax.errorbar(row["x"], row["y"],
                    yerr=[[row["y"] - row["lo"]],
                          [row["hi"] - row["y"]]],
                    fmt="none", ecolor="#111111", capsize=2, linewidth=1.1)

    all_rows = [row for row in old if row["x"] <= 0.35] + new
    y_lo = min(row["lo"] for row in all_rows)
    y_hi = max(row["hi"] for row in all_rows)
    ax.axhline(0.3, color="#a46a1f", linestyle="--", linewidth=1,
               label="Low-rho CI boundary")
    ax.annotate(MATCHED_NOTES[name] + r" $\rightarrow$",
                xy=(0.97, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=7.5, color="#666666")
    ax.set_title(name)
    ax.set_xlim(0, 0.32)
    ax.set_xlabel("Achieved teacher-forced per-step KL")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    return y_lo, y_hi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(NEW_RUN / "rho_vs_absolute_kl.png"),
    )
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    extents = [plot_panel(ax, name) for ax, name in zip(axes, ["FV", "Task vector"])]
    y_lo = min(extent[0] for extent in extents)
    # Keep the 0.3 low-rho boundary line inside the axes so its legend entry
    # is never dead.
    y_hi = max(max(extent[1] for extent in extents), 0.32)
    pad = 0.08 * (y_hi - y_lo)
    axes[0].set_ylim(y_lo - pad, y_hi + pad)
    axes[0].set_ylabel("rho: fraction of native behavioral effect reproduced")
    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=3,
               frameon=False, fontsize=8)
    fig.suptitle("Output-interface frontiers on the absolute-KL axis", fontsize=12)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(out)
    print(out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
