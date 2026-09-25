# tests/test_readme.py
"""The quickstart in the README and on the docs home page must run (BN-230).

The README is also the project page on PyPI, and its quickstart had stopped
working: it built its own small data provider, and the engine had since
started asking providers for more than that provider offered. Nothing ran the
example, so nothing noticed. These tests run it, exactly as a reader would
copy it.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def quickstart(path: Path) -> str:
    """The first Python block under the Quickstart heading."""
    text = path.read_text(encoding="utf-8")
    section = text[text.index("uickstart"):]

    return re.search(r"```python\n(.*?)```", section, re.S).group(1)


@pytest.mark.parametrize("page", ["README.md", "docs/index.md"])
def test_the_quickstart_runs_as_written(tmp_path,
                                        page):
    script = tmp_path / "quickstart.py"
    script.write_text(quickstart(ROOT / page), encoding="utf-8")

    completed = subprocess.run([sys.executable, str(script)],
                               capture_output=True,
                               text=True,
                               cwd=tmp_path,
                               timeout=300,
                               check=False)

    assert completed.returncode == 0, completed.stderr
    assert "Final index level:" in completed.stdout


def test_both_pages_show_the_same_example():
    """One example, not two copies drifting apart."""
    assert quickstart(ROOT / "README.md") == quickstart(ROOT / "docs/index.md")
