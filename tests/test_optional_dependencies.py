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
GUARDED_MODULE = "yfinance"
GUARDED_EXTRA = "data"


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
        assert f'pip install "py-beacon-kit[{GUARDED_EXTRA}]"' in message

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
    # Not reached through `beacon.data`, and the one module a spawned server
    # needs before anything else works — so it has to load with nothing
    # installed. `platformdirs` is imported inside `default_path()` for exactly
    # this reason (BN-113).
    "beacon.data.store",
    "beacon.derivatives",
    "beacon.fund",
    "beacon.index",
    # The derived-index calculation (BN-167) solves with scipy at solve time
    # only; the module itself must import on the core install. Its level
    # chaining (BN-171) is pure pandas and must stay that way.
    "beacon.index.chaining",
    "beacon.index.derived",
    # The optimise package imports scipy-free (BN-166): constraints, configs
    # and results are descriptions, and only an actual solve needs scipy. The
    # submodules are listed individually as well as through the package so a
    # future import cannot hide behind __init__'s import order.
    "beacon.optimise",
    "beacon.optimise.config",
    "beacon.optimise.constraints",
    "beacon.optimise.result",
    "beacon.portfolio",
    # Charts, interactive charts, PDF reports, the solver and Postgres stores
    # joined the core in BN-310, so they import with only the extras blocked.
    "beacon.plot",
    "beacon.plot.paint",
    "beacon.plot.interactive.figure",
    "beacon.report.pdf",
    "beacon.optimise.solver",
    "beacon.data.postgres",
    # The generator is pandas and numpy only (BN-114); its CLI needs
    # `platformdirs` solely to resolve a default output path.
    "beacon.synthetic",
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
    """The core pipeline must import with only its core dependencies present.

    Those are pandas, numpy, pydantic and — since BN-180 — `exchange_calendars`,
    which is no longer blocked here because it is no longer optional. Every
    other optional package still is, so the test says exactly what it used to:
    nothing in the core reaches for an extra.
    """

    def test_exchange_calendars_is_no_longer_optional(self):
        """The guard's own registry is the statement of what is optional.

        Left in `EXTRA_FOR_MODULE` it would keep promising an extra that
        pyproject no longer declares, and `require()` would still be the way to
        import it — which is the thing BN-180 stopped doing.
        """
        assert "exchange_calendars" not in EXTRA_FOR_MODULE

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


class TestTheCoreInstall:
    """What used to be extras is core: the packages are declared as core
    dependencies, and the old extras stay declared, empty, so an install
    command written for an earlier release still works."""

    CORE = ("matplotlib", "fonttools", "plotly", "reportlab", "openpyxl", "scipy",
            "psycopg[binary]", "platformdirs")
    FOLDED = ("excel", "postgres", "pdf", "optimise", "plot", "plot-interactive")

    @staticmethod
    def project() -> dict:
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("pyproject.toml not available (installed-only checkout)")

        import tomllib

        with pyproject.open("rb") as handle:
            return tomllib.load(handle)["project"]

    def test_charts_reports_excel_the_optimiser_and_postgres_are_core(self):
        declared = [entry.split(">")[0].split("<")[0]
                    for entry in self.project()["dependencies"]]

        assert [name for name in self.CORE if name not in declared] == []

    def test_the_folded_extras_stay_declared_and_empty(self):
        extras = self.project()["optional-dependencies"]

        assert {name: extras.get(name) for name in self.FOLDED} == {
            name: [] for name in self.FOLDED}

    def test_none_of_them_is_guarded_any_more(self):
        guarded = {"matplotlib", "fontTools", "plotly", "reportlab", "openpyxl", "scipy",
                   "psycopg", "platformdirs"}

        assert guarded.isdisjoint(EXTRA_FOR_MODULE)


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

    def test_the_calendar_package_is_declared_as_a_core_dependency(self):
        """BN-180. The behaviour is proved elsewhere; this is the declaration.

        An environment that happened to have `exchange_calendars` installed for
        some other reason would hide a missing entry, and the first person to
        hit it would be a user of a released wheel rather than this suite.
        """
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("pyproject.toml not available (installed-only checkout)")

        import tomllib

        with pyproject.open("rb") as handle:
            project = tomllib.load(handle)["project"]

        assert "exchange_calendars" in project["dependencies"]
        assert "calendars" not in project["optional-dependencies"], (
            "the `calendars` extra was removed when the package became core; "
            "leaving it declared would promise an install that means nothing")


class TestTheServerExtraIsSelfSufficient:
    """`pip install "py-beacon-kit[server]"` must give a server that starts.

    The server mounts the optimiser and report routers, which import scipy and
    reportlab at module scope. Both are core, so the extra need bring only
    the server's own packages.
    """

    def test_the_app_builds_with_only_the_server_stack(self):
        """The other extra (data) is blocked, so a new router quietly
        importing yfinance fails here."""
        script = BARE_ENVIRONMENT_SCRIPT.replace(
            "for name in {modules!r}:\n    importlib.import_module(name)",
            "from beacon.server import ServerConfig, create_app\n"
            "app = create_app(ServerConfig(auth_token='x'))\n"
            "assert len(app.openapi()['paths']) > 30")

        completed = subprocess.run(
            [sys.executable, "-c", script.format(blocked=["yfinance"], modules=[])],
            capture_output=True, text=True, check=False)

        assert completed.returncode == 0, completed.stderr
        assert "ok" in completed.stdout

    def test_the_extra_declares_what_the_server_imports(self):
        """The declaration, not just the behaviour."""
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("pyproject.toml not available (installed-only checkout)")

        import tomllib

        with pyproject.open("rb") as handle:
            server = tomllib.load(handle)["project"]["optional-dependencies"]["server"]

        assert {"fastapi", "orjson", "websockets"} <= set(server)
        assert any(entry.startswith("uvicorn") for entry in server)
