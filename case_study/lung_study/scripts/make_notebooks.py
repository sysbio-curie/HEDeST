"""
Builds the notebooks from their sources in ``notebooks/src`` (percent format: ``# %%`` starts
a code cell, ``# %% [markdown]`` a markdown cell whose lines start with ``# ``).

``00_precheck.py`` is a template, instantiated once per sample ({{SAMPLE}}); every other source
gives one notebook of the same name.

Usage: python scripts/make_notebooks.py [name ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

STUDY = Path(__file__).resolve().parents[1]
SOURCES = STUDY / "notebooks" / "src"
SAMPLES = ["A_AAAA1879", "A_AAAA1638", "A_AAAA1665"]


def cells_of(text: str) -> list:
    cells, kind, lines = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(lines).strip("\n")
        if kind == "markdown":
            body = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(body))
        elif body.strip():
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        else:
            lines.append(line)
    flush()

    return cells


def write(name: str, text: str) -> Path:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = cells_of(text)
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (hedest-env)", "language": "python"}
    path = STUDY / "notebooks" / f"{name}.ipynb"
    nbformat.write(notebook, path)
    print(f"wrote {path.relative_to(STUDY)}")

    return path


if __name__ == "__main__":
    wanted = sys.argv[1:]
    setup = (SOURCES / "_setup.txt").read_text().strip("\n")
    for source in sorted(SOURCES.glob("*.py")):
        text = source.read_text().replace("{{SETUP}}", setup)
        if source.stem == "00_precheck":
            for sample in SAMPLES:
                name = f"00_precheck_{sample}"
                if not wanted or name in wanted or source.stem in wanted:
                    write(name, text.replace("{{SAMPLE}}", sample))
        elif not wanted or source.stem in wanted:
            write(source.stem, text)
