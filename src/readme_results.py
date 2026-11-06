"""Rewrite the README results block from saved comparison files.

If `results/tables/main_comparison.csv` is missing, the block says results are
pending. Numbers are never typed in by hand.
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import config as C
from .utils import get_logger

logger = get_logger(__name__)

START = "<!-- RESULTS_START -->"
END = "<!-- RESULTS_END -->"
README_PATH = C.ROOT / "README.md"


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_results_block() -> str:
    """Build the markdown that sits between the results markers."""
    comparison = C.TABLES_DIR / "main_comparison.csv"
    verdict = C.TABLES_DIR / "verdict.md"
    if not comparison.exists():
        logger.info("main_comparison.csv is absent; writing the pending results block")
        return "Results pending.\n"
    frame = pd.read_csv(comparison)
    parts = [
        "Numbers below are read from `results/tables/main_comparison.csv`.",
        "",
        _markdown_table(frame),
        "",
    ]
    if verdict.exists():
        parts.append(verdict.read_text(encoding="utf-8").strip())
        parts.append("")
    else:
        parts.append("Pending: waiting for results/tables/verdict.md")
        parts.append("")
    return "\n".join(parts)


def update_readme(text: str | None = None) -> str:
    """Replace the marked results block. Returns the new README text."""
    original = README_PATH.read_text(encoding="utf-8") if text is None else text
    if START not in original or END not in original:
        raise RuntimeError(f"{README_PATH} is missing {START} or {END}")
    before, rest = original.split(START, 1)
    _, after = rest.split(END, 1)
    updated = f"{before}{START}\n{render_results_block()}{END}{after}"
    if text is None:
        README_PATH.write_text(updated, encoding="utf-8")
        logger.info("Updated results block in %s", README_PATH)
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject saved results into README.md.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    update_readme()


if __name__ == "__main__":
    main()
