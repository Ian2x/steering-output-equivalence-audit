#!/usr/bin/env python3
"""Generate compact paper-summary figures from the frozen result ledger."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures"


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def rho_kappa_map() -> None:
    # Values are frozen in results.md's master table (Table 3 of the paper).
    # Rows are sorted by rho point estimate, descending. kappa kinds:
    #   measured      -> filled marker with paired-bootstrap 95% CI
    #   construction  -> open marker at 1.0 (prefill-only native schedule)
    #   point_only    -> filled marker, no interval shipped
    rows = [
        ("Output-push anchor (A0)", 1.615, 1.250, 2.185, 0.000, 0.000, 0.000, "measured"),
        ("CAA corrigibility", 1.333, 0.933, 2.000, 0.833, 0.538, 1.067, "measured"),
        ("Activation Addition", 0.959, 0.853, 1.071, 1.000, None, None, "construction"),
        ("CAA sycophancy", 0.882, 0.790, 0.971, 0.833, 0.750, 0.913, "measured"),
        ("SAE steering", 0.416, 0.336, 0.493, 0.631, 0.553, 0.707, "measured"),
        ("Refusal ablation", 0.259, 0.180, 0.338, 0.986, 0.957, 1.000, "measured"),
        ("Function vector", -0.119, -0.231, -0.039, 0.983, None, None, "point_only"),
        ("Task vector", -0.141, -0.250, -0.061, 1.000, None, None, "construction"),
    ]
    ink = "#222222"
    ys = np.arange(len(rows))
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.4, 3.6), sharey=True,
        gridspec_kw={"width_ratios": [1.45, 1]},
    )

    # Left panel: rho forest with the preregistered summary bands.
    from matplotlib.transforms import blended_transform_factory
    ax.axvspan(-0.40, 0.30, color="#e8f1f8", alpha=0.8, zorder=0)
    ax.axvspan(0.90, 2.30, color="#f8ece9", alpha=0.8, zorder=0)
    band_trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(0.30, 0.02, "0.3", fontsize=7.5, color="#365b75", ha="center",
            va="bottom", transform=band_trans)
    ax.text(0.90, 0.02, "0.9", fontsize=7.5, color="#8c4a3f", ha="center",
            va="bottom", transform=band_trans)
    ax.axvline(0, color="#888888", lw=0.7, zorder=1)
    for y, (name, rho, lo, hi, *_rest) in zip(ys, rows):
        if name == "Refusal ablation":
            # NLL-exempt plateau maximum with its exact transformed
            # interval; the strict-gate maximum 0.151 is ticked separately.
            # Open marker: a gate-convention maximum, not a plain measurement.
            ax.errorbar(rho, y, xerr=[[rho - lo], [hi - rho]], fmt="o",
                        color=ink, capsize=2.5, markersize=5.5,
                        markerfacecolor="white", markeredgewidth=1.2, zorder=3)
            ax.plot([0.151], [y], marker="|", markersize=9, color=ink,
                    markeredgewidth=1.4, zorder=3)
            ax.annotate("strict 0.151", (0.151, y), xytext=(-4, 9),
                        textcoords="offset points", fontsize=7, color="#444444")
        else:
            ax.errorbar(rho, y, xerr=[[rho - lo], [hi - rho]], fmt="o",
                        color=ink, capsize=2.5, markersize=5.5,
                        markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    ax.set_yticks(ys, [row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(-0.40, 2.30)
    ax.set_xlabel("Fraction of native effect reproduced (rho)")
    ax.set_title("Output reproducibility, sorted", fontsize=10)
    ax.grid(axis="x", color="#e3e3e3", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    # Right panel: kappa on the same row order, which visibly does not
    # follow the rho ordering.
    for y, (name, _rho, _lo, _hi, kappa, klo, khi, kind) in zip(ys, rows):
        if kind == "construction":
            ax2.plot([kappa], [y], marker="o", markersize=5.5,
                     markerfacecolor="white", markeredgecolor=ink,
                     markeredgewidth=1.2, zorder=3)
        elif kind == "point_only":
            ax2.plot([kappa], [y], marker="o", markersize=5.5, color=ink,
                     markeredgecolor="white", markeredgewidth=0.7, zorder=3)
            ax2.annotate("point only", (kappa, y), xytext=(-46, -3),
                         textcoords="offset points", fontsize=7, color="#444444")
        else:
            ax2.errorbar(kappa, y, xerr=[[kappa - klo], [khi - kappa]],
                         fmt="o", color=ink, capsize=2.5, markersize=5.5,
                         markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    ax2.set_xlim(-0.06, 1.12)
    ax2.set_xlabel("First-token intervention share (kappa)")
    ax2.set_title("kappa does not follow the rho order", fontsize=10)
    ax2.grid(axis="x", color="#e3e3e3", lw=0.6)
    ax2.spines[["top", "right"]].set_visible(False)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", linestyle="", color=ink,
               markeredgecolor="white", label="measured, 95% CI"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor=ink,
               label="by construction, or a gate-convention maximum"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "rho_kappa_map")


def caa_semantic_robustness() -> None:
    labels = ["Sycophancy", "Corrigibility-match"]
    means = np.array([
        [59.85, 69.675, 77.65],
        [31.90, 45.40, 48.70],
    ])
    conditions = ["Base", "Native CAA", "Output control"]
    colors = ["#b3b3b3", "#2563a6", "#c44e52"]
    x = np.arange(len(labels))
    width = 0.23
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={"width_ratios": [1.15, 1]})
    handles = []
    for j, condition in enumerate(conditions):
        handles.append(ax.bar(x + (j - 1) * width, means[:, j], width, label=condition,
                              color=colors[j], edgecolor="white", linewidth=0.7))
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Cleaned automated endorsement (0-100)")
    ax.set_title("Pointwise semantic metric (means)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", lw=0.6)

    # Paired effects with 95% CIs from semantic_judged.json; rows ordered so
    # sycophancy is on top, matching the left panel's left-to-right order.
    effects = {
        # behavior -> [(label, point, lo, hi, color)]
        "Sycophancy": [
            ("native - base", 9.825, 3.549, 16.125, "#2563a6"),
            ("control - base", 17.8, 12.7, 23.0, "#c44e52"),
            ("control - native", 7.975, 2.30, 13.975, "#222222"),
        ],
        "Corrigibility-match": [
            ("native - base", 13.5, 6.9, 20.4, "#2563a6"),
            ("control - base", 16.8, 9.2, 24.4, "#c44e52"),
            ("control - native", 3.30, -3.20, 9.90, "#222222"),
        ],
    }
    span = ax2.axvspan(-10, 10, color="#e8f1f8", alpha=0.8, zorder=0)
    ax2.axvline(0, color="#666666", lw=0.8)
    group_centers = []
    y = 0.0
    for behavior in labels:
        group_start = y
        for (name, point, lo, hi, color) in effects[behavior]:
            ax2.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o",
                         color=color, capsize=2.5, markersize=4.8,
                         markeredgecolor="white", markeredgewidth=0.6, zorder=3)
            y += 1.0
        group_centers.append((group_start + y - 1.0) / 2)
        y += 0.7
    ax2.set_yticks(group_centers, labels)
    ax2.invert_yaxis()
    ax2.set_xlim(-15, 27)
    ax2.set_xlabel("Paired difference in cleaned score")
    ax2.set_title("Paired effects, 95% CIs")
    ax2.grid(axis="x", color="#e0e0e0", lw=0.6)
    ax2.spines[["top", "right"]].set_visible(False)

    from matplotlib.lines import Line2D
    effect_handles = [
        Line2D([], [], marker="o", linestyle="", color="#2563a6", label="native - base"),
        Line2D([], [], marker="o", linestyle="", color="#c44e52", label="control - base"),
        Line2D([], [], marker="o", linestyle="", color="#222222", label="control - native"),
        span,
    ]
    fig.legend(handles=effect_handles,
               labels=["native - base", "control - base", "control - native",
                       "equivalence margin (control - native)"],
               frameon=False, fontsize=6.8, ncol=2, loc="lower right",
               bbox_to_anchor=(0.99, 0.015))
    fig.suptitle("CAA output controls survive an automated semantic check, but both cells remain Mixed", y=1.02)
    fig.legend([handle[0] for handle in handles], conditions, frameon=False,
               loc="upper center", bbox_to_anchor=(0.30, 0.90), ncol=3, fontsize=7.5)
    fig.text(0.01, 0.015,
             "GPT-4.1-mini pointwise metric with surface-only\nresponses zeroed; not human semantic or\ndistributional equivalence.",
             fontsize=7, color="#444444")
    fig.tight_layout(rect=(0, 0.13, 1, 0.88))
    save(fig, "caa_semantic_robustness")


def main() -> None:
    style()
    rho_kappa_map()
    caa_semantic_robustness()


if __name__ == "__main__":
    main()
