# tests/test_docs_examples.py
"""Every Python example in the docs runs, as a reader would copy it.

Each page's ```python blocks run in order as one script, so a later block can
use what an earlier one defined, in a fresh interpreter and an empty folder.
A block that must not run (one that starts a server, say) is marked with an
HTML comment on the line before it, giving the reason:

    <!-- not run: starts a server and waits for requests -->
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

BLOCK = re.compile(r"(?:<!-- not run: (?P<reason>[^>]+?) -->\s*\n)?"
                   r"```python\n(?P<code>.*?)```", re.S)


def examples(page: Path) -> list[str]:
    """The runnable Python blocks on a page, in order."""
    text = page.read_text(encoding="utf-8")

    return [match.group("code") for match in BLOCK.finditer(text)
            if match.group("reason") is None]


def pages() -> list[Path]:
    found = [page for page in sorted(DOCS.rglob("*.md"))
             if "reference" not in page.parts and examples(page)]

    return [ROOT / "README.md", *found]


@pytest.mark.parametrize("page", pages(),
                         ids=lambda page: str(page.relative_to(ROOT)))
def test_the_examples_run(tmp_path,
                          page):
    script = tmp_path / "example.py"
    script.write_text("\n".join(examples(page)), encoding="utf-8")

    completed = subprocess.run([sys.executable, str(script)],
                               capture_output=True,
                               text=True,
                               cwd=tmp_path,
                               env={**os.environ, "MPLBACKEND": "Agg"},
                               timeout=600,
                               check=False)

    assert completed.returncode == 0, completed.stderr[-3000:]


def test_a_skipped_block_says_why():
    """The marker needs a reason, so a skip is a decision, not a shortcut."""
    for page in DOCS.rglob("*.md"):
        text = page.read_text(encoding="utf-8")

        assert "<!-- not run -->" not in text, page
