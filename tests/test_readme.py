# tests/test_readme.py
"""The quickstart in the README and on the docs home page must run (BN-230).

The README is also the project page on PyPI, and its quickstart had stopped
working: it built its own small data provider, and the engine had since
started asking providers for more than that provider offered. Nothing ran the
example, so nothing noticed. These tests run it, exactly as a reader would
copy it.

The quickstart is a run of notebook-style cells, and the pages show the
tables the cells produce (BN-287). Those tables are checked against a fresh
run, so a change to the engine cannot leave the pages showing old numbers.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["README.md", "docs/index.md"]

# Appended to the cells: the two tables the pages show, as CSV.
PRINT_TABLES = """
print("DAILY")
print(daily.tail().round(2).to_csv(header=False, date_format="%Y-%m-%d"))
print("SUMMARY")
print(pd.DataFrame({"Portfolio": result.summary()}).round(4).to_csv(header=False))
"""


def section(path: Path) -> str:
    """The page's Quickstart section, up to the next heading."""
    text = path.read_text(encoding="utf-8")
    start = text.index("## Quickstart")

    return text[start:text.index("\n## ", start + 1)]


def quickstart(path: Path) -> str:
    """Every Python cell under the Quickstart heading, in order."""
    return "\n".join(re.findall(r"```python\n(.*?)```", section(path), re.S))


def shown_tables(path: Path) -> list[list[str]]:
    """The rows of the Markdown tables in the Quickstart section, as cells."""
    rows = [line.strip("|").split("|") for line in section(path).splitlines()
            if line.startswith("| ") and not line.startswith("| |")]

    return [[cell.strip() for cell in row] for row in rows]


def run(page: str,
        tmp_path: Path) -> str:
    script = tmp_path / "quickstart.py"
    script.write_text(quickstart(ROOT / page) + PRINT_TABLES, encoding="utf-8")

    completed = subprocess.run([sys.executable, str(script)],
                               capture_output=True,
                               text=True,
                               cwd=tmp_path,
                               env={**os.environ, "MPLBACKEND": "Agg"},
                               timeout=300,
                               check=False)

    assert completed.returncode == 0, completed.stderr

    return completed.stdout


@pytest.mark.parametrize("page", PAGES)
def test_the_quickstart_runs_and_shows_what_it_produces(tmp_path,
                                                         page):
    output = run(page, tmp_path)
    printed = [line.split(",") for line in output.splitlines() if "," in line]

    assert "SUMMARY" in output

    def values(row: list[str]) -> tuple:
        return (row[0], *(round(float(cell), 4) for cell in row[1:]))

    assert [values(row) for row in shown_tables(ROOT / page)] == [
        values(row) for row in printed]


def test_both_pages_show_the_same_example():
    """One example, not two copies drifting apart."""
    assert quickstart(ROOT / "README.md") == quickstart(ROOT / "docs/index.md")
    assert shown_tables(ROOT / "README.md") == shown_tables(ROOT / "docs/index.md")
