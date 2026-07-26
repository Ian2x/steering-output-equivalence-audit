#!/usr/bin/env python3
"""Fetch primary arXiv metadata for the manuscript bibliography."""

from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


IDS = [
    "2306.03341", "2308.10248", "2310.01405", "2310.15213",
    "2310.15916", "2312.06681", "2406.05946", "2406.11717",
    "2410.13928", "2410.17245", "2501.17148", "2502.02716", "2507.11878",
    "2509.06608", "2602.01716", "2602.06801", "2604.02608",
    "2604.09839", "2604.15557", "2604.23178", "2605.03907",
    "2605.05983", "2605.07990",
]

OUT = Path(__file__).resolve().parents[1] / "paper_references.bib"
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def braces(text: str) -> str:
    return text.replace("{", r"\{").replace("}", r"\}")


def main() -> None:
    query = urllib.parse.urlencode({
        "id_list": ",".join(IDS),
        "max_results": len(IDS),
    })
    with urllib.request.urlopen(
            f"https://export.arxiv.org/api/query?{query}", timeout=30) as response:
        root = ET.fromstring(response.read())

    records = {}
    for entry in root.findall("a:entry", ATOM):
        arxiv_id = entry.find("a:id", ATOM).text.rsplit("/", 1)[-1].split("v")[0]
        title = " ".join(entry.find("a:title", ATOM).text.split())
        authors = " and ".join(
            node.find("a:name", ATOM).text
            for node in entry.findall("a:author", ATOM))
        year = entry.find("a:published", ATOM).text[:4]
        category = entry.find("a:category", ATOM).attrib["term"]
        records[arxiv_id] = (title, authors, year, category)

    missing = sorted(set(IDS) - set(records))
    if missing:
        raise RuntimeError(f"missing arXiv records: {missing}")

    blocks = ["% Generated from the primary arXiv API by exp/fetch_paper_bib.py."]
    for arxiv_id in IDS:
        title, authors, year, category = records[arxiv_id]
        key = "arxiv" + arxiv_id.replace(".", "_")
        blocks.extend([
            f"@misc{{{key},",
            f"  title = {{{braces(title)}}},",
            f"  author = {{{braces(authors)}}},",
            f"  year = {{{year}}},",
            f"  eprint = {{{arxiv_id}}},",
            "  archivePrefix = {arXiv},",
            f"  primaryClass = {{{category}}},",
            f"  url = {{https://arxiv.org/abs/{arxiv_id}}}",
            "}",
            "",
        ])
    OUT.write_text("\n".join(blocks), encoding="utf-8")
    print(f"wrote {OUT} ({len(records)} records)")


if __name__ == "__main__":
    main()
