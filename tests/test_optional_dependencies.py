# tests/test_optional_dependencies.py
"""Unit tests for beacon._optional — the optional-dependency import guards."""
import importlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from beacon._optional import EXTRA_FOR_MODULE, require
from beacon.exceptions import BeaconError, MissingDependencyError

# A registered optional module used as the subject of the guard tests. Which
# one does not matter — the import itself is always faked, so these tests give
# the same answer whether or not the extra happens to be installed.
GUARDED_MODULE = "openpyxl"
GUARDED_EXTRA = "excel"


def fake_import(raises: bool):
    """Build an import_module stand-in that either fails or returns a module."""
    def _import(name: str) -> ModuleType:
        if raises:
            raise ImportError(f"No module named '{name}'")
        return ModuleType(name)

    return _import


class TestRequireSuccess:

    def test_returns_the_imported_module(self,
                                         monkeypatch):
        monkeypatch.setattr(importlib, "import_module", fake_import(raises=False))

        module = require(GUARDED_MODULE, "Excel reporting")

        assert isinstance(module, ModuleType)
        assert module.__name__ == GUARDED_MODULE

    def test_unregistered_module_is_a_programming_error(self):
        with pytest.raises(KeyError):
            require("not_a_registered_dependency", "Something")


class TestRequireFailure:

    @pytest.fixture(autouse=True)
    def _break_imports(self,
                       monkeypatch):
        monkeypatch.setattr(importlib, "import_module", fake_import(raises=True))

    def test_raises_missing_dependency_error(self):
        with pytest.raises(MissingDependencyError):
            require(GUARDED_MODULE, "Excel reporting")

    def test_message_names_the_feature_module_and_extra(self):
        with pytest.raises(MissingDependencyError) as excinfo:
            require(GUARDED_MODULE, "Excel reporting")

        message = str(excinfo.value)

        assert "Excel reporting" in message
        assert GUARDED_MODULE in message
        assert f'pip install "py-beacon[{GUARDED_EXTRA}]"' in message

    def test_carries_structured_fields(self):
        with pytest.raises(MissingDependencyError) as excinfo:
            require(GUARDED_MODULE, "Excel reporting")

        error = excinfo.value

        assert error.module_name == GUARDED_MODULE
        assert error.feature == "Excel reporting"
        assert error.extra == GUARDED_EXTRA

    def test_is_catchable_as_import_error_and_beacon_error(self):
        with pytest.raises(ImportError):
            require(GUARDED_MODULE, "Excel reporting")

        with pytest.raises(BeaconError):
            require(GUARDED_MODULE, "Excel reporting")

    def test_chains_the_original_import_error(self):
        with pytest.raises(MissingDependencyError) as excinfo:
            require(GUARDED_MODULE, "Excel reporting")

        assert isinstance(excinfo.value.__cause__, ImportError)


CORE_MODULES = [
    "beacon",
    "beacon.asset",
    "beacon.backtest",
    "beacon.data",
    "beacon.derivatives",
    "beacon.fund",
    "beacon.index",
    "beacon.portfolio",
    "beacon.testing",
    "beacon.tokens",
]

# Imports every core module in a fresh interpreter where the optional packages
# have been made unimportable, proving the core needs none of them. Run as a
# subprocess because the block has to be in place before the first import.
BARE_ENVIRONMENT_SCRIPT = """
import importlib
import sys

BLOCKED = {blocked!r}


class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ImportError(f"{{fullname}} is blocked for this test")
        return None


sys.meta_path.insert(0, Blocker())

for name in {modules!r}:
    importlib.import_module(name)

print("ok")
"""


class TestCoreImportsAreDependencyFree:
    """The core pipeline must import with only pandas and numpy present."""

    def test_core_imports_without_any_optional_dependency(self):
        script = BARE_ENVIRONMENT_SCRIPT.format(
            blocked=sorted(EXTRA_FOR_MODULE),
            modules=CORE_MODULES)

        completed = subprocess.run([sys.executable, "-c", script],
                                   capture_output=True,
                                   text=True,
                                   check=False)

        assert completed.returncode == 0, (
            f"core import pulled in an optional dependency:\n{completed.stderr}")
        assert "ok" in completed.stdout


class TestGuardedFeatures:
    """Features behind an extra report the extra rather than failing obscurely."""

    def test_excel_report_reports_the_missing_extra(self,
                                                    monkeypatch):
        from beacon.portfolio import ReportGenerator

        monkeypatch.setattr(importlib, "import_module", fake_import(raises=True))

        with pytest.raises(MissingDependencyError,
                           match=r'py-beacon\[excel\]'):
            ReportGenerator().generate_performance_report_excel(
                performance_data=None,
                report_path="unused.xlsx")


    def test_the_optimiser_reports_the_missing_extra(self):
        """Importing beacon.optimise without scipy names the extra to install.

        Run in a subprocess with scipy blocked, because the guard fires at
        import time and this interpreter has already imported the package.
        """
        script = BARE_ENVIRONMENT_SCRIPT.format(blocked=["scipy"],
                                                modules=["beacon.optimise"])

        completed = subprocess.run([sys.executable, "-c", script],
                                   capture_output=True,
                                   text=True,
                                   check=False)

        assert completed.returncode != 0
        assert 'py-beacon[optimise]' in completed.stderr
        assert "MissingDependencyError" in completed.stderr


class TestExtrasAreDeclared:
    """Every extra referenced by the guard must exist in pyproject.toml."""

    def test_each_mapped_extra_is_declared(self):
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("pyproject.toml not available (installed-only checkout)")

        import tomllib

        with pyproject.open("rb") as handle:
            declared = tomllib.load(handle)["project"]["optional-dependencies"]

        for module_name, extra in EXTRA_FOR_MODULE.items():
            assert extra in declared, (
                f"'{module_name}' maps to extra '{extra}', which pyproject.toml "
                f"does not declare")


class TestTheServerExtraIsSelfSufficient:
    """`pip install "py-beacon[server]"` must give a server that starts.

    The server mounts the optimiser and report routers, which import scipy and
    reportlab at module scope, so the `server` extra has to bring those too.
    This broke the OpenAPI export job in CI, which installed only `[server]`
    and could not build the app.
    """

    def test_the_app_builds_with_only_the_server_stack(self):
        """Everything outside the server's own dependency set is blocked, so a
        new router quietly importing an unrelated extra fails here."""
        script = BARE_ENVIRONMENT_SCRIPT.replace(
            "for name in {modules!r}:\n    importlib.import_module(name)",
            "from beacon.server import ServerConfig, create_app\n"
            "app = create_app(ServerConfig(auth_token='x'))\n"
            "assert len(app.openapi()['paths']) > 30")

        completed = subprocess.run(
            [sys.executable, "-c",
             script.format(blocked=["yfinance", "matplotlib", "openpyxl", "plotly"],
                           modules=[])],
            capture_output=True, text=True, check=False)

        assert completed.returncode == 0, completed.stderr
        assert "ok" in completed.stdout

    def test_the_extra_declares_what_the_server_imports(self):
        """The declaration, not just the behaviour: an environment that happens
        to have scipy installed would hide a missing entry."""
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("pyproject.toml not available (installed-only checkout)")

        import tomllib

        with pyproject.open("rb") as handle:
            server = tomllib.load(handle)["project"]["optional-dependencies"]["server"]

        assert "py-beacon[optimise]" in server
        assert "py-beacon[pdf]" in server
