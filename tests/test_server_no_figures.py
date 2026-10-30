# tests/test_server_no_figures.py
"""BN-401: the server computes no figures. It turns requests into library
calls and results into JSON, so a figure has one definition, the library's,
whether it is read from Python or from the API.

Read from the source rather than by running it: no module under
``beacon/server`` calls a numeric reduction or transform (returns, sums,
means, spreads, logs), and the views that turn a run into figures do no
arithmetic at all.
"""
import ast
from pathlib import Path

import pytest

import beacon.server

SERVER = Path(beacon.server.__file__).parent

# Numeric methods and functions: pandas' and numpy's reductions and
# transforms, and math's and numpy's functions.
NUMERIC = {"pct_change", "diff", "cumprod", "cumsum", "cummax", "cummin", "prod", "sum",
           "mean", "median", "std", "var", "cov", "corr", "quantile", "rolling",
           "expanding", "ewm", "log", "log1p", "exp", "expm1", "sqrt", "power"}

# Pricing a derivative is still worked out here; it moves to the library with
# the holdings and market cap views' arithmetic under BN-417.
PENDING = {"derivatives.py"}

# The views that read a run's figures: no arithmetic of their own.
VIEWS = ("periods.py", "comparisons.py", "views.py", "weights.py", "index_level.py",
         "risk.py", "optimisation.py", "share_class_results.py")
ARITHMETIC = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow)


def modules() -> list[Path]:
    return sorted(SERVER.rglob("*.py"))


def numeric_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))

    return [f"{path.name}:{node.lineno} {node.func.attr}" for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in NUMERIC]


def arithmetic(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))

    return [f"{path.name}:{node.lineno} {ast.unparse(node)}" for node in ast.walk(tree)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ARITHMETIC)]


class TestNoFigures:

    def test_no_module_calls_a_numeric_reduction_or_transform(self):
        found = [call for path in modules() if path.name not in PENDING
                 for call in numeric_calls(path)]

        assert found == []

    @pytest.mark.parametrize("name", VIEWS)
    def test_the_views_do_no_arithmetic(self,
                                        name):
        assert arithmetic(SERVER / name) == []

    def test_the_check_sees_a_figure(self,
                                     tmp_path):
        source = tmp_path / "view.py"
        source.write_text("def f(prices):\n    returns = prices.pct_change()\n"
                          "    return returns.iloc[-1] / returns.iloc[0] - 1\n",
                          encoding="utf-8")

        assert numeric_calls(source) == ["view.py:2 pct_change"]
        assert len(arithmetic(source)) == 2

    def test_every_view_named_exists(self):
        assert all((SERVER / name).exists() for name in VIEWS)
