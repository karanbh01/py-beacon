"""Tests for the derivatives package skeleton."""

import importlib
from pathlib import Path

from beacon.derivatives import *  # noqa: F401,F403


def test_derivatives_package_structure_exists():
    root = Path("beacon/derivatives")

    assert root.is_dir()
    for module in ["__init__.py", "base.py", "pricing.py", "futures.py", "swaps.py", "forwards.py"]:
        assert (root / module).is_file()


def test_derivatives_package_and_modules_import_without_circular_errors():
    import beacon.derivatives  # noqa: F401

    for module_name in ["base", "pricing", "futures", "swaps", "forwards"]:
        importlib.import_module(f"beacon.derivatives.{module_name}")
