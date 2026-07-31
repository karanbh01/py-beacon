# src/beacon/plot/__init__.py
"""
Charting.

Reached through a `.plot` accessor on the result objects — `IndexResult`,
`BacktestResult`, `AttributionResult`, `OptimisationResult`, `RiskModel` —
rather than through free functions taking a result. That follows the same
stance as the asset views: plotting is a *result-layer* concern, so a result
knows how to draw itself and nothing else grows a chart method.

    result.plot.level()
    backtest.plot.performance()

Needs matplotlib, which ships in the `plot` extra:

    pip install "py-beacon[plot]"

**Importing `beacon` costs nothing extra.** The accessor is a descriptor that
resolves on first access, so matplotlib is imported when a chart is drawn and
not before. `import beacon` is as fast without matplotlib installed as it was
before this package existed, and a test asserts it.

Every method takes `ax=` and returns the `Axes` it drew on, so charts compose
into a figure the caller laid out. The signatures carry nothing
matplotlib-specific, which is what keeps an interactive backend possible later
without changing any call site.
"""
from .base import ChartMethods, PlotAccessor

__all__ = [
    "ChartMethods",
    "PlotAccessor",
    "compare",
    "palette",
    "style",
    "use",
]


def __getattr__(name: str) -> object:
    """Resolve the matplotlib-dependent names lazily.

    Module-level `__getattr__` so `beacon.plot.compare` and `beacon.plot.style`
    work as attributes without importing matplotlib at package import. A plain
    import at the top would defeat the whole arrangement.

    Resolved through `import_module` rather than `from . import style`: the
    latter looks the name up on this package first, which re-enters this
    function and recurses until the stack runs out.
    """
    import importlib

    if name == "compare":
        return importlib.import_module(".comparison", __name__).compare

    if name in ("style", "use", "palette"):
        module = importlib.import_module(".style", __name__)

        return module if name == "style" else getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
