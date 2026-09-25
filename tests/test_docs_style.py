# tests/test_docs_style.py
"""The published docs follow the house style, and the reference is complete.

pybeacon.dev renders three things from this repository: the pages in docs/,
the Python reference (from docstrings), and the server API (from the OpenAPI
spec). All three are for users, so none of them carries an em-dash or an
internal ticket number. See tests/docstyle.py.
"""
import json
import re

import pytest

from beacon.server import ServerConfig, create_app
from docstyle import (
    DOCS,
    EM_DASH,
    TICKET,
    docstring_problems,
    page_problems,
    public_modules,
)

REFERENCE = DOCS / "reference"
MKDOCS = DOCS.parent / "mkdocs.yml"
DIRECTIVE = re.compile(r"^::: (beacon[\w.]*)", re.M)


def test_the_pages_follow_the_house_style():
    problems = list(page_problems())

    assert not problems, "\n".join(problems)


def test_public_docstrings_follow_the_house_style():
    problems = list(docstring_problems())

    assert not problems, "\n".join(problems)


def test_the_api_spec_follows_the_house_style():
    spec = json.dumps(create_app(ServerConfig(auth_token="t")).openapi(),
                      ensure_ascii=False)

    assert EM_DASH not in spec
    assert not TICKET.findall(spec)


class TestTheReferenceIsComplete:

    @pytest.fixture(scope="class")
    def documented(self) -> set[str]:
        """Every module a reference page renders directly."""
        found: set[str] = set()

        for page in REFERENCE.rglob("*.md"):
            found.update(DIRECTIVE.findall(page.read_text(encoding="utf-8")))

        return found

    def test_pages_render_their_submodules(self):
        """So a package's page covers every module inside it."""
        assert "show_submodules: true" in MKDOCS.read_text(encoding="utf-8")

    def test_every_public_module_has_a_page(self,
                                            documented):
        """Its own, or its package's. The top-level `beacon` only re-exports
        names that the Derivatives and Data sources pages render."""
        missing = [name for name in public_modules()
                   if name != "beacon" and not any(name == page or name.startswith(f"{page}.")
                              for page in documented)]

        assert not missing, "\n".join(missing)

    def test_every_page_is_in_the_nav(self):
        nav = MKDOCS.read_text(encoding="utf-8")
        pages = [f"reference/{page.name}" for page in REFERENCE.glob("*.md")]

        assert [page for page in pages if page not in nav] == []
