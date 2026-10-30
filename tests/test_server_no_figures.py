# tests/test_server_no_figures.py
"""BN-401, BN-417: the server computes nothing. It turns requests into
library calls and results into JSON, so a figure has one definition, the
library's, whether it is read from Python or from the API.

Read from the source rather than by running it. No module under
``beacon/server`` calls a numeric reduction or transform (returns, sums,
means, spreads, logs), and every piece of arithmetic left in it is listed
below with why it is not a figure: joining text, lists and sets, counting,
positions in a list, a job's progress, file paths, and checks that a number
is finite. Anything else fails, and belongs in the library.
"""
import ast
from pathlib import Path

import beacon.server

SERVER = Path(beacon.server.__file__).parent

# Numeric methods and functions: pandas' and numpy's reductions and
# transforms, and math's and numpy's functions.
NUMERIC = {"pct_change", "diff", "cumprod", "cumsum", "cummax", "cummin", "prod", "sum",
           "mean", "median", "std", "var", "cov", "corr", "quantile", "rolling",
           "expanding", "ewm", "log", "log1p", "exp", "expm1", "sqrt", "power"}

# numpy and math functions that check or reshape rather than calculate.
CHECKS = {"isfinite", "isnan", "isinf", "flatnonzero", "unique", "array"}

OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.Mod)
BUILTINS = {"sum", "round", "abs", "min", "max", "divmod", "pow"}

# A path joined with `/`: its left side names a folder.
FOLDERS = ("folder", "directory", "staging", "base", "saved", "managed_root")

TEXT = "joins text"
COLLECTIONS = "joins or takes away lists, tuples or sets"
COUNTING = "counts or positions in a list"
PROGRESS = "a job's progress"
BOOKKEEPING = "a store's own bookkeeping"

ALLOWED = {
    ("analysis.py", "reason[:1].upper() + reason[1:]"): TEXT,
    ("backtest_options.py", "len(specs) + 1"): COUNTING,
    ("backtest_options.py", "len(specs) + 2"): COUNTING,
    ("backtest_options.py", "len(specs) + 3"): COUNTING,
    ("backtest_options.py", "len(specs) + 4"): COUNTING,
    ("backtest_options.py", "prefix + parameter.name"): TEXT,
    ("backtest_settings.py", "COMMON_SETTINGS + (ETF_SETTINGS if exchange_traded else ())"):
        COLLECTIONS,
    ("backtest_settings.py", "set(spec.params) - set(parameters)"): COLLECTIONS,
    ("backtests.py", "len(from_capital) - len(level)"): COUNTING,
    ("backtests.py", "RUN_START + RUN_SPAN * fraction"): PROGRESS,
    ("backtests.py", "RUN_SPAN * fraction"): PROGRESS,
    ("constraints.py", "set(row.params) - accepted"): COLLECTIONS,
    ("data_stores.py", "suffix += 1"): COUNTING,
    ("data_stores.py", "max(active, key=lambda record: record['last_loaded_at'] or '')"):
        BOOKKEEPING,
    ("data_stores.py", "sum((item.stat().st_size for item in path.iterdir() if item.is_file()))"):
        BOOKKEEPING,
    ("definitions.py", "set(spec_params) - allowed"): COLLECTIONS,
    ("documents.py", "self.unparseable + self.from_newer_build + self.unrecognised"): COUNTING,
    ("documents.py", "self.unparseable + self.from_newer_build"): COUNTING,
    ("documents.py", "causes[cause] += 1"): COUNTING,
    ("holdings.py", "stages += implementation"): COLLECTIONS,
    ("holdings.py", "left -= {name for name, by in plan['removed'].items() if by == kind}"):
        COLLECTIONS,
    ("holdings.py", "position - 1"): COUNTING,
    ("jobs.py", "max(matching, key=lambda doc: str(doc.get('completed_at', '')))"): BOOKKEEPING,
    ("jobs.py", "forgotten += 1"): COUNTING,
    ("jobs.py", "len(history) - MAX_STORED_RESULTS"): COUNTING,
    ("jobs.py", "round(self.progress, 6)"): PROGRESS,
    ("jobs.py", "self._last_completed + timedelta(microseconds=1)"): BOOKKEEPING,
    ("jobs.py", "min(max(progress, 0.0), 1.0)"): PROGRESS,
    ("jobs.py", "max(progress, 0.0)"): PROGRESS,
    ("preview.py", "max(date, pd.Timestamp(definition.source.base_date))"): BOOKKEEPING,
    ("preview.py", "max(applicable)"): BOOKKEEPING,
    ("preview.py", "position - 1"): COUNTING,
    ("preview.py", "step.position - 1"): COUNTING,
    ("record_tables.py", "FUND_WEIGHTS + HOLDINGS"): COLLECTIONS,
    ("record_tables.py", "min(total, MAX_WEIGHT_DATES)"): COUNTING,
    ("record_tables.py", "len(lookup) - 1"): COUNTING,
    ("reference.py", "parts += [part.strip() for part in value.split(',') if part.strip()]"):
        COLLECTIONS,
    ("reference.py", "sum((1 for entry in entries if not entry.found))"): COUNTING,
    ("reference.py", "requested - derived_requested"): COLLECTIONS,
    ("reference.py", "set(columns) - set(available)"): COLLECTIONS,
    ("routers/data.py", "offset + limit"): COUNTING,
    ("routers/data.py", "len(unknown) - len(shown)"): COUNTING,
    ("routers/indices.py", "position += 1"): COUNTING,
    ("routers/jobs.py", "live + stored"): COLLECTIONS,
    ("routers/refresh.py", "REFRESH_SHARE * done / total"): PROGRESS,
    ("routers/refresh.py", "REFRESH_SHARE * done"): PROGRESS,
    ("routers/refresh.py", "last + timedelta(days=1)"): BOOKKEEPING,
    ("routers/synthetic.py", "float(fraction) * GENERATION_SHARE"): PROGRESS,
    ("routers/universes.py", "len(missing) - len(shown)"): COUNTING,
    ("schemas.py", "EQUITISE_NUMBERS + EQUITISE_FLAGS + EQUITISE_TEXTS"): COLLECTIONS,
    ("schemas.py", "EQUITISE_NUMBERS + EQUITISE_FLAGS"): COLLECTIONS,
    ("stage_descriptions.py", "params += values.rows()"): COLLECTIONS,
    ("stage_descriptions.py", "'at most ' + values.text('holdings')"): TEXT,
    ("store.py", "version += 1"): BOOKKEEPING,
}

# Text and lists joined across lines, matched by how they start.
ALLOWED_STARTS = {
    ("methods.py", "'^' + re.sub("): TEXT,
    ("routers/stores.py", 'f"The store \'{record[\'name\']}\' cannot be read: " + '): TEXT,
    ("routers/synthetic.py", "f'the generator exited with code {process.returncode}: ' + "): TEXT,
    ("stage_descriptions.py", "([f\"cuts each position to {' and to '.join(cuts)}\"] if cuts "
                              "else []) + "): COLLECTIONS,
    ("stage_descriptions.py", "(clauses or [\"holds each position within the vehicle's "
                              "limits\"]) + "): COLLECTIONS,
}


def modules() -> list[Path]:
    return sorted(SERVER.rglob("*.py"))


def name_of(path: Path) -> str:
    return path.relative_to(SERVER).as_posix() if path.is_relative_to(SERVER) else path.name


def numeric_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))

    return [f"{name_of(path)}:{node.lineno} {node.func.attr}" for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in NUMERIC]


def arithmetic(path: Path) -> list[str]:
    """Every operator, built-in, or numpy or math function that calculates
    in *path*, less path joins."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, OPERATORS):
            if not (isinstance(node.op, ast.Div)
                    and any(word in ast.unparse(node.left) for word in FOLDERS)):
                found.append(node)
        elif ((isinstance(node, ast.AugAssign) and isinstance(node.op, OPERATORS))
              or (isinstance(node, ast.Call) and _calculates(node.func))):
            found.append(node)

    return [ast.unparse(node) for node in found]


def _calculates(func: ast.expr) -> bool:
    if isinstance(func, ast.Name):
        return func.id in BUILTINS

    return (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
            and func.value.id in ("math", "np", "numpy") and func.attr not in CHECKS)


def allowed(module: str,
            expression: str) -> bool:
    return ((module, expression) in ALLOWED
            or any(module == where and expression.startswith(start)
                   for where, start in ALLOWED_STARTS))


class TestNoFigures:

    def test_no_module_calls_a_numeric_reduction_or_transform(self):
        assert [call for path in modules() for call in numeric_calls(path)] == []

    def test_every_calculation_left_is_listed_as_not_a_figure(self):
        unlisted = [f"{name_of(path)}: {expression}" for path in modules()
                    for expression in arithmetic(path)
                    if not allowed(name_of(path), expression)]

        assert unlisted == []

    def test_everything_listed_is_still_there(self):
        """So the list shrinks with the code rather than going stale."""
        present = {(name_of(path), expression) for path in modules()
                   for expression in arithmetic(path)}
        stale = [key for key in ALLOWED if key not in present]
        stale += [key for key in ALLOWED_STARTS
                  if not any(module == key[0] and expression.startswith(key[1])
                             for module, expression in present)]

        assert stale == []

    def test_the_check_sees_a_figure(self,
                                     tmp_path):
        source = tmp_path / "view.py"
        source.write_text("import math\n"
                          "def f(prices, folder):\n"
                          "    returns = prices.pct_change()\n"
                          "    path = folder / 'x.json'\n"
                          "    total = sum(returns)\n"
                          "    return returns.iloc[-1] / returns.iloc[0] - 1, math.expm1(total)\n",
                          encoding="utf-8")

        assert numeric_calls(source) == ["view.py:3 pct_change", "view.py:6 expm1"]
        assert sorted(arithmetic(source)) == sorted(["sum(returns)",
                                                     "returns.iloc[-1] / returns.iloc[0] - 1",
                                                     "returns.iloc[-1] / returns.iloc[0]",
                                                     "math.expm1(total)"])
