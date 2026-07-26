#!/usr/bin/env python3
"""DEPRECATED: draw the rho-kappa paper figure with the standard library only.

This generator emitted a PDF referencing non-embedded Type 1 Helvetica, which
renders with invisible text in pdf.js/pdfium viewers. The shipped figure is
now produced by plot_paper_summary.py (matplotlib, TrueType fonts embedded).
Kept for the historical record; refuses to run so the two generators cannot
diverge again.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.exit(
    "draw_rho_kappa_pdf.py is deprecated: use plot_paper_summary.py, which "
    "embeds fonts. This script is retained only for the historical record."
)


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper_figures" / "rho_kappa_map.pdf"

PAGE_W, PAGE_H = 511.2, 302.4
LEFT, RIGHT, BOTTOM, TOP = 58.0, 496.0, 43.0, 268.0
X_MIN, X_MAX = -0.35, 2.25
Y_MIN, Y_MAX = -0.04, 1.10

ROWS = [
    ("Output-push anchor", 1.615, 1.250, 2.185, 0.000, "#6b7280", 6, 8),
    ("Function vector", -0.119, -0.231, -0.039, 0.9830508474576272, "#2563a6", 7, -13),
    ("Activation Addition", 0.959, 0.853, 1.071, 1.000, "#c44e52", 6, 8),
    ("Task vector", -0.141, -0.250, -0.061, 1.000, "#2563a6", 7, 9),
    ("SAE steering", 0.416, 0.336, 0.493, 0.631, "#2a8f62", 6, 8),
    ("CAA sycophancy", 0.882, 0.790, 0.971, 0.8333333333333333, "#c44e52", -77, -13),
    ("CAA corrigibility", 1.333, 0.933, 2.000, 0.8333333333333331, "#c44e52", 7, 8),
]


def rgb(hex_color: str) -> tuple[float, float, float]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))


def px(value: float) -> float:
    return LEFT + (value - X_MIN) / (X_MAX - X_MIN) * (RIGHT - LEFT)


def py(value: float) -> float:
    return BOTTOM + (value - Y_MIN) / (Y_MAX - Y_MIN) * (TOP - BOTTOM)


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class Canvas:
    def __init__(self) -> None:
        self.ops: list[str] = []

    def color(self, value: str, *, stroke: bool = False) -> None:
        vals = rgb(value)
        self.ops.append(
            f"{vals[0]:.4f} {vals[1]:.4f} {vals[2]:.4f} "
            + ("RG" if stroke else "rg")
        )

    def line(self, x1: float, y1: float, x2: float, y2: float,
             *, width: float = 0.6, color: str = "#000000") -> None:
        self.color(color, stroke=True)
        self.ops.append(f"{width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def rect(self, x: float, y: float, w: float, h: float,
             *, fill: str) -> None:
        self.color(fill)
        self.ops.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re f")

    def circle(self, x: float, y: float, radius: float, *, fill: str) -> None:
        k = 0.5522847498 * radius
        self.color(fill)
        self.ops.append(
            f"{x + radius:.2f} {y:.2f} m "
            f"{x + radius:.2f} {y + k:.2f} {x + k:.2f} {y + radius:.2f} "
            f"{x:.2f} {y + radius:.2f} c "
            f"{x - k:.2f} {y + radius:.2f} {x - radius:.2f} {y + k:.2f} "
            f"{x - radius:.2f} {y:.2f} c "
            f"{x - radius:.2f} {y - k:.2f} {x - k:.2f} {y - radius:.2f} "
            f"{x:.2f} {y - radius:.2f} c "
            f"{x + k:.2f} {y - radius:.2f} {x + radius:.2f} {y - k:.2f} "
            f"{x + radius:.2f} {y:.2f} c f"
        )
        self.color("#ffffff", stroke=True)
        self.ops.append(
            f"0.7 w {x + radius:.2f} {y:.2f} m "
            f"{x + radius:.2f} {y + k:.2f} {x + k:.2f} {y + radius:.2f} "
            f"{x:.2f} {y + radius:.2f} c "
            f"{x - k:.2f} {y + radius:.2f} {x - radius:.2f} {y + k:.2f} "
            f"{x - radius:.2f} {y:.2f} c "
            f"{x - radius:.2f} {y - k:.2f} {x - k:.2f} {y - radius:.2f} "
            f"{x:.2f} {y - radius:.2f} c "
            f"{x + k:.2f} {y - radius:.2f} {x + radius:.2f} {y - k:.2f} "
            f"{x + radius:.2f} {y:.2f} c S"
        )

    def triangle_left(self, x: float, y: float, size: float, *,
                      stroke: str) -> None:
        self.color("#ffffff")
        self.ops.append(
            f"{x - size:.2f} {y:.2f} m {x + size:.2f} {y + size:.2f} l "
            f"{x + size:.2f} {y - size:.2f} l h f"
        )
        self.color(stroke, stroke=True)
        self.ops.append(
            f"1.1 w {x - size:.2f} {y:.2f} m "
            f"{x + size:.2f} {y + size:.2f} l "
            f"{x + size:.2f} {y - size:.2f} l h S"
        )

    def text(self, x: float, y: float, text: str, *, size: float = 7.0,
             bold: bool = False, color: str = "#222222",
             rotate: float = 0.0) -> None:
        self.color(color)
        font = "F2" if bold else "F1"
        angle = math.radians(rotate)
        a, b = math.cos(angle), math.sin(angle)
        c, d = -b, a
        self.ops.append(
            f"BT /{font} {size:.2f} Tf "
            f"{a:.5f} {b:.5f} {c:.5f} {d:.5f} {x:.2f} {y:.2f} Tm "
            f"({pdf_escape(text)}) Tj ET"
        )


def write_pdf(path: Path, stream: bytes) -> None:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W:.1f} {PAGE_H:.1f}] "
            "/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> "
            "/Contents 4 0 R >>"
        ).encode(),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
        + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ]
    data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode())
        data.extend(obj)
        data.extend(b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    data.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        (
            f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main() -> None:
    c = Canvas()
    c.rect(px(-0.35), BOTTOM, px(0.30) - px(-0.35), TOP - BOTTOM, fill="#e8f1f8")
    c.rect(px(0.30), BOTTOM, px(0.90) - px(0.30), TOP - BOTTOM, fill="#f2f2f2")
    c.rect(px(0.90), BOTTOM, px(2.25) - px(0.90), TOP - BOTTOM, fill="#f8ece9")

    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        c.line(LEFT, py(tick), RIGHT, py(tick), color="#dddddd", width=0.45)
        c.text(39, py(tick) - 2.3, f"{tick:.1f}", size=6.5, color="#555555")
    for tick in [-0.25, 0.0, 0.5, 1.0, 1.5, 2.0]:
        c.line(px(tick), BOTTOM, px(tick), BOTTOM - 3, color="#555555")
        c.text(px(tick) - 6, BOTTOM - 13, f"{tick:g}", size=6.5, color="#555555")

    c.line(LEFT, BOTTOM, RIGHT, BOTTOM, color="#555555")
    c.line(LEFT, BOTTOM, LEFT, TOP, color="#555555")
    c.line(px(0.0), BOTTOM, px(0.0), TOP, color="#888888", width=0.55)

    for name, point, lo, hi, kappa, color, dx, dy in ROWS:
        y = py(kappa)
        c.line(px(lo), y, px(hi), y, color=color, width=0.85)
        c.line(px(lo), y - 2.4, px(lo), y + 2.4, color=color, width=0.85)
        c.line(px(hi), y - 2.4, px(hi), y + 2.4, color=color, width=0.85)
        if name == "Task vector":
            c.triangle_left(px(point), y, 3.6, stroke=color)
        else:
            c.circle(px(point), y, 3.3, fill=color)
        c.text(px(point) + dx, y + dy, name, size=6.9, color="#222222")

    refusal_x, refusal_y = px(0.259), py(0.9856115107913669)
    c.triangle_left(refusal_x, refusal_y, 3.6, stroke="#8c564b")
    c.text(refusal_x + 7, refusal_y - 14, "Refusal ablation", size=6.9)

    c.text(128, 286, "Temporal sufficiency and output reproducibility are distinct",
           size=10.2, bold=True, color="#222222")
    c.text(188, 16, "Behavioral effect fraction reproduced (rho)",
           size=7.6, color="#333333")
    c.text(14, 112, "First-token intervention share (kappa)",
           size=7.6, color="#333333", rotate=90)
    c.text(
        8, 3,
        "Intervals are paired-bootstrap 95% CIs. Refusal rho <= 0.259 and is gate-sensitive. "
        "Shading shows preregistered summaries, not mechanism classes.",
        size=5.7, color="#444444",
    )
    write_pdf(PDF, "\n".join(c.ops).encode("latin-1"))
    print(PDF)


if __name__ == "__main__":
    main()

