# tests/test_migration_guide.py
"""BN-464: the guide to moving to 0.11.0 lists every 0.10.0 path a result
warns for, with the replacement the warning names, so the two cannot drift."""
from pathlib import Path

import pytest

from beacon.backtest.result import MOVED, MOVED_METHODS

GUIDE = Path(__file__).resolve().parent.parent / "docs" / "moving-to-0.11.md"


@pytest.fixture(scope="module")
def guide() -> str:
    return GUIDE.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", sorted(MOVED))
def test_each_moved_name_is_listed_with_its_home(guide,
                                                 name):
    home = MOVED[name][0]

    assert f"`result.{name}`" in guide
    assert f"result.{home}" in guide


@pytest.mark.parametrize("name", sorted(MOVED_METHODS))
def test_each_moved_method_is_listed_with_its_home(guide,
                                                   name):
    path = MOVED_METHODS[name][0].split(",")[0]

    assert f"`result.{name}()`" in guide
    assert f"result.{path}" in guide


def test_the_renamed_names_are_listed(guide):
    for old, new in (("Book", "PortfolioRecord"), ("IndexBooks", "IndexRecords"),
                     ("ShareClassBook", "ShareClassRecord"), ("BOOKS", "PORTFOLIOS"),
                     ("book=", "of=")):
        assert f"| `{old}`" in guide and f"`{new}`" in guide


def test_it_is_in_the_nav():
    nav = (GUIDE.parent.parent / "mkdocs.yml").read_text(encoding="utf-8")

    assert "moving-to-0.11.md" in nav
