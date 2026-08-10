# tests/test_examples.py
"""BN-129: every example runs.

An example that no longer works is worse than no example — it looks
authoritative and teaches something false — and it breaks silently unless
something executes it. So each script is run end to end in a subprocess, which
is how a reader runs it, rather than importing `main()` and hoping the module
scope is the same thing.

The assertions are deliberately about *shape*: exit code, that the expected
sections were printed, and that a few numbers are present and finite. Pinning
exact figures would make every example a change-detector for the generator.
"""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

SCRIPTS = sorted(path for path in EXAMPLES.glob("*.py")
                 if not path.name.startswith("_"))

# Each script and a phrase it must print. Not a full transcript — just enough
# that a script which silently degraded to printing nothing is caught.
EXPECTED = {
    "01_index_and_backtest.py": ("Methodology", "Backtest", "Index versus portfolio"),
    "02_backtest_analysis.py": ("Summary statistics", "Attribution", "residual"),
    "03_index_futures.py": ("Fair value", "Basis", "Rolling to the next contract"),
    "04_optimised_index.py": ("Risk model", "Binding constraints", "Solution"),
    "05_optimised_backtest.py": ("Optimisation", "Side by side",
                                 "Ex ante versus realised"),
}


# Each script is run **once** and its result shared across the assertions
# below. Running per assertion tripled the wall clock for no extra coverage —
# fifteen subprocesses to learn what five could tell us.
_RESULTS: dict[str, subprocess.CompletedProcess] = {}


def run(script: Path) -> subprocess.CompletedProcess:
    """Run one example the way a reader would, once per session."""
    if script.name not in _RESULTS:
        _RESULTS[script.name] = subprocess.run(
            [sys.executable, str(script), "--quiet"],
            capture_output=True, text=True, check=False,
            cwd=script.parent, timeout=900)

    return _RESULTS[script.name]


def test_every_example_is_covered():
    """A script added without an entry here would never be checked."""
    assert {path.name for path in SCRIPTS} == set(EXPECTED)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_it_runs(script):
    completed = run(script)

    assert completed.returncode == 0, (
        f"{script.name} exited {completed.returncode}\n"
        f"--- stdout ---\n{completed.stdout[-3000:]}\n"
        f"--- stderr ---\n{completed.stderr[-3000:]}")


@pytest.mark.timeout(900)
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_it_prints_its_sections(script):
    """A script that runs but reports nothing teaches nothing."""
    completed = run(script)

    for phrase in EXPECTED[script.name]:
        assert phrase in completed.stdout, (
            f"{script.name} did not print {phrase!r}")


@pytest.mark.timeout(900)
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_it_prints_no_broken_numbers(script):
    """`nan` and `inf` on screen are the failure a shape assertion misses: the
    script runs, prints its sections, and every figure is meaningless."""
    completed = run(script)
    lowered = completed.stdout.lower()

    for token in ("nan%", " nan", "inf%", " inf", "-inf"):
        assert token not in lowered, (
            f"{script.name} printed {token!r}:\n{completed.stdout[-2000:]}")


@pytest.mark.timeout(900)
def test_the_output_is_ascii():
    """Printed output has to survive a Windows console.

    The file is UTF-8 and its prose can say what it likes, but an em dash
    inside a `print()` renders as a replacement character on cp1252 — which is
    what happened first time round.
    """
    import ast

    def non_ascii(script) -> list[str]:
        tree = ast.parse(script.read_text(encoding="utf-8"))

        return [f"{script.name}:{node.lineno}"
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "print"
                for piece in ast.walk(node)
                if isinstance(piece, ast.Constant)
                and isinstance(piece.value, str)
                and not piece.value.isascii()]

    offenders = [found for script in [*SCRIPTS, EXAMPLES / "_shared.py"]
                 for found in non_ascii(script)]

    assert not offenders, f"non-ASCII inside print(): {offenders}"


class TestTheyStandAlone:
    """The properties that make an example runnable from a clean checkout."""

    def test_none_of_them_need_a_prepared_store(self):
        """Each generates its own data. One that read the app-data store would
        pass here and fail on a machine that had never run the generator."""
        for script in SCRIPTS:
            source = script.read_text(encoding="utf-8")

            assert "default_path" not in source, script.name
            assert "store.load" not in source, script.name

    def test_they_share_their_setup(self):
        """Five copies of the same fixture is five things to update."""
        for script in SCRIPTS:
            assert "_shared" in script.read_text(encoding="utf-8"), script.name
