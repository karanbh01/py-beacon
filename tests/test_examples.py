# tests/test_examples.py
"""BN-129: every example notebook runs.

An example that no longer works is worse than no example — it looks
authoritative and teaches something false — and it breaks silently unless
something executes it. So each notebook is run end to end through a real
Jupyter kernel, which is how a reader runs it, rather than extracting the code
cells and `exec`-ing them. Extraction would pass while the notebook itself was
unopenable, and would silently drop every trailing-expression display, which is
most of what these notebooks show.

The assertions are deliberately about *shape*: that it executes, that the
sections a reader is promised are present, and that no figure came out `nan`.
Pinning exact numbers would make every notebook a change-detector for the
generator.
"""
import json
import re
from pathlib import Path

import pytest

# The whole module runs five real Jupyter kernels through full pipelines --
# 3-7 minutes, the single largest cost in the suite. Routine local runs skip
# it (-m "not slow", or scripts/test_chunks.sh); CI and pre-push runs keep it.
pytestmark = pytest.mark.slow

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))

# Each notebook and a heading it must contain. Not a full transcript — just
# enough that one which silently degraded to a title page is caught.
EXPECTED = {
    "01_index_and_backtest.ipynb": ("define the methodology",
                                    "calculate the index",
                                    "the gap between the two"),
    "02_backtest_analysis.ipynb": ("summary statistics",
                                   "attribution",
                                   "the reconciliation"),
    "03_index_futures.ipynb": ("fair value",
                               "basis against a quoted price",
                               "rolling to the next contract"),
    "04_optimised_index.ipynb": ("the risk model",
                                 "which constraints actually bound",
                                 "where the risk actually sits"),
    "05_optimised_backtest.ipynb": ("optimise at *every* rebalance",
                                    "ex ante versus realised",
                                    "what the constraints cost"),
}

# Executing five notebooks through a kernel is the slowest thing in the suite,
# so each runs **once** and the executed copy is shared across the assertions
# below. Running per assertion tripled the wall clock for no extra coverage.
_EXECUTED: dict[str, object] = {}


def executed(path: Path):
    """Run one notebook the way a reader would, once per session."""
    if path.name not in _EXECUTED:
        notebook = nbformat.read(path, as_version=4)

        nbclient.NotebookClient(
            notebook, timeout=1200, kernel_name="python3",
            resources={"metadata": {"path": str(path.parent)}}).execute()

        _EXECUTED[path.name] = notebook

    return _EXECUTED[path.name]


def outputs(notebook) -> list[str]:
    """Every piece of text a notebook printed or displayed."""
    found = []

    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            text = output.get("text") or output.get("data", {}).get("text/plain")

            if text:
                found.append("".join(text) if isinstance(text, list) else text)

    return found


def sources(notebook,
            kind: str) -> str:
    """All source of one cell type, lowercased, joined."""
    return "\n".join(cell.source for cell in notebook.cells
                     if cell.cell_type == kind).lower()


def test_every_notebook_is_covered():
    """One added without an entry here would never be checked."""
    assert {path.name for path in NOTEBOOKS} == set(EXPECTED)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_it_runs(path):
    """The whole point. `execute()` raises on the first cell that fails."""
    assert executed(path) is not None


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_it_has_its_sections(path):
    """A notebook that runs but explains nothing teaches nothing."""
    prose = sources(executed(path), "markdown")

    for heading in EXPECTED[path.name]:
        assert heading in prose, f"{path.name} is missing {heading!r}"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_it_produces_no_broken_numbers(path):
    """`nan` and `inf` on screen are the failure a shape assertion misses: the
    notebook runs, prints its sections, and every figure is meaningless.

    Matched on word boundaries rather than as substrings. A plain `in` check
    fires on "Financials" and "financing", both of which these notebooks print
    legitimately -- and a check that cries wolf gets deleted rather than fixed.
    """
    broken = re.compile(r"(?<![\w.])[-+]?(nan|inf)(?![\w.])", re.IGNORECASE)

    for text in outputs(executed(path)):
        found = broken.search(text)

        assert found is None, (
            f"{path.name} produced {found.group()!r}:\n{text[:2000]}")


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_its_code_output_is_ascii(path):
    """Printed output has to survive a Windows console.

    Markdown cells render in a browser and may say what they like, but a
    notebook converted to a script or run through a terminal frontend prints
    its `print()` calls to cp1252 — where an em dash becomes a replacement
    character, which is what happened when these were scripts.
    """
    for cell in executed(path).cells:
        if cell.cell_type != "code":
            continue

        for line in cell.source.splitlines():
            assert line.isascii(), f"{path.name}: non-ASCII code: {line!r}"


class TestTheyStandAlone:
    """The properties that make a notebook runnable from a clean checkout."""

    @pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
    def test_it_needs_no_prepared_store(self, path):
        """Each generates its own data. One that read the app-data store would
        pass here and fail on a machine that had never run the generator."""
        source = path.read_text(encoding="utf-8")

        assert "default_path" not in source, path.name
        assert "store.load" not in source, path.name

    @pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
    def test_it_imports_no_local_module(self, path):
        """A notebook is opened and run on its own, often from a directory
        that is not this one. Sharing a helper module across them -- which is
        what these did as scripts -- makes every one of them unopenable
        anywhere else, so the duplicated setup is deliberate.
        """
        source = path.read_text(encoding="utf-8")

        assert "_shared" not in source, path.name
        assert "from helpers" not in source, path.name

    @pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
    def test_it_is_committed_without_outputs(self, path):
        """Stored outputs make every rerun a diff of megabytes of base64.

        Checked against the file on disk, not the executed copy in memory.
        """
        notebook = json.loads(path.read_text(encoding="utf-8"))

        offenders = [cell.get("id") for cell in notebook["cells"]
                     if cell.get("outputs") or cell.get("execution_count")]

        assert not offenders, (
            f"{path.name} has stored output in {offenders}; "
            f"clear it before committing")
