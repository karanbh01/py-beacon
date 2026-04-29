"""Tests for derivatives public API metadata wiring."""

from pathlib import Path

import beacon
from beacon.derivatives import IndexFuture, TotalReturnSwap


def test_derivatives_exports_are_available_from_public_api():
    assert beacon.derivatives.IndexFuture is IndexFuture
    assert IndexFuture.__name__ == "IndexFuture"
    assert TotalReturnSwap.__name__ == "TotalReturnSwap"


def test_project_metadata_mentions_derivatives_and_version_bumped():
    pyproject = Path("pyproject.toml").read_text()

    assert 'version = "0.0.2"' in pyproject
    assert 'description = "End-to-end toolkit for index, ETF, and delta-one derivatives development"' in pyproject
    for keyword in ["derivatives", "futures", "swaps", "delta-one"]:
        assert f'"{keyword}"' in pyproject


def test_readme_module_list_mentions_delta_one_derivatives():
    assert "5. Delta-1 Derivatives" in Path("README.md").read_text()
