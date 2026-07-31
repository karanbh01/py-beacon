# scripts/build_gallery.py
"""
Render every chart kind and write the documentation gallery.

Run at docs build time, so the gallery shows what the code currently draws
rather than screenshots someone took once. A gallery that can go stale is worse
than no gallery: it becomes a record of what the charts used to look like, and
nobody knows which images are still true.

Every chart is drawn in both styles from `beacon.testing.dataset`, so the images
are reproducible — the same run on any machine produces the same pictures, for
the same reason the image-regression baselines can exist at all.

Usage:

    python scripts/build_gallery.py [output_directory]

Defaults to `docs/gallery`.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from beacon.analysis import attribute, drifted_weights
from beacon.backtest.engine import BacktestEngine
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import MarketCapWeighted
from beacon.optimise import (
    FullInvestment,
    GroupBounds,
    PositionBounds,
    minimise_tracking_error,
)
from beacon.optimise.frontier import efficient_frontier
from beacon.plot import compare, use
from beacon.risk import estimate_risk_model
from beacon.testing import dataset
from beacon.tokens import DARK, LIGHT

DEFAULT_OUTPUT = Path("docs/gallery")

END = "2024-12-31"
CAP = 0.20
RISK_FREE = 0.02

# Screen resolution rather than print: these are viewed in a browser, and a
# 300dpi page would be four times the bytes for no visible gain.
DPI = 110

# Chart id -> what it is for, shown as the caption. Written here rather than
# scraped from docstrings because a caption and an API docstring answer
# different questions.
CAPTIONS = {
    "level": "Index level, rebased to 100, against an optional benchmark.",
    "weights": "Constituent weights at a rebalance, with the cap marked.",
    "performance": "Growth of 100 with a linked drawdown panel.",
    "annual_returns": "Calendar-year returns as signed bars.",
    "contributions": "Per-constituent contribution to return, with the drags.",
    "compare": "Several results on one axis, rebased on their shared window.",
    "exposures": "Active weights against the index, sign-coloured.",
    "frontier": "The efficient frontier, its named points and the capital "
                "market line.",
    "correlation": "Correlation matrix on the beacon_corr scale.",
}


def build_results() -> dict[str, object]:
    """Everything the gallery draws from, computed once."""
    fetcher = dataset.data_fetcher()

    definition = IndexDefinition(
        index_id="CANON", index_name="Canonical Index",
        base_date=dataset.START, base_value=1000.0, currency="USD",
        eligibility_rules=[], weighting_scheme=MarketCapWeighted(),
        rebalancing_frequency="QUARTERLY",
        universe_identifiers=list(dataset.UNIVERSE),
        max_constituent_weight=CAP)

    index_result = IndexCalculator(definition, fetcher).run(
        start_date=dataset.START, end_date=END)

    backtest = BacktestEngine(
        start_date=dataset.START, end_date=END, initial_capital=10_000_000.0,
        data_provider=fetcher, target_index_result=index_result,
        transaction_cost_bps=10.0).run()

    prices = dataset.prices().loc[:END]
    weights = drifted_weights(index_result.weight_snapshots, prices)
    asset_returns = prices.pct_change().reindex(weights.index)
    period_returns = (weights.shift(1) * asset_returns).sum(axis=1)
    attribution = attribute(period_returns, weights, asset_returns,
                            cap_drag=-0.003, cost_drag=-0.004)

    risk_model = estimate_risk_model(dataset.returns(), intensity=0.1)

    optimisation = minimise_tracking_error(
        dataset.equal_weights(),
        [FullInvestment(), PositionBounds(0.0, 0.25),
         GroupBounds("Technology", dataset.sectors()["Technology"],
                     maximum=CAP)],
        risk_model)

    expected = {name: 0.04 + 0.02 * position
                for position, name in enumerate(dataset.UNIVERSE)}
    frontier = efficient_frontier(risk_model, expected, points=12,
                                  risk_free_rate=RISK_FREE)

    return {"index": index_result, "backtest": backtest,
            "attribution": attribution, "risk": risk_model,
            "optimisation": optimisation, "frontier": frontier,
            "benchmark": prices["CCC"]}


def chart_calls(results: dict[str, object]) -> dict[str, object]:
    """Chart id to a callable that draws it."""
    index = results["index"]
    backtest = results["backtest"]

    return {
        "level": lambda: index.plot.level(benchmark=results["benchmark"]),
        "weights": lambda: index.plot.weights(),
        "performance": lambda: backtest.plot.performance(),
        "annual_returns": lambda: backtest.plot.annual_returns(),
        "contributions": lambda: results["attribution"].plot.contributions(),
        "compare": lambda: compare(index, backtest),
        "exposures": lambda: results["optimisation"].plot.exposures(),
        "frontier": lambda: results["optimisation"].plot.frontier(
            results["frontier"], risk_free_rate=RISK_FREE),
        "correlation": lambda: results["risk"].plot.correlation(),
    }


def render_all(directory: Path) -> list[str]:
    """Draw every chart in both styles.

    Returns:
        list: The chart ids rendered, in gallery order.
    """
    directory.mkdir(parents=True, exist_ok=True)

    results = build_results()
    calls = chart_calls(results)

    for mode in (LIGHT, DARK):
        use(mode)
        suffix = "" if mode == LIGHT else "-dark"

        for name, call in calls.items():
            call()
            plt.savefig(directory / f"{name}{suffix}.png", dpi=DPI)
            plt.close("all")

    use(LIGHT)

    return list(calls)


def write_page(directory: Path,
               names: list[str]) -> Path:
    """Write the gallery's markdown page.

    Both modes are shown side by side rather than one behind a toggle: the
    point of having two styles is that a reader can see whether the dark one
    actually works, and a toggle hides exactly that comparison.
    """
    lines = [
        "# Chart gallery",
        "",
        "Every chart Beacon draws, rendered from the canonical synthetic",
        "dataset in both styles. Generated at docs build time, so these are",
        "what the code currently produces rather than screenshots taken once.",
        "",
        "Light and dark are shown together on purpose: the point of having two",
        "styles is that a reader can see whether the dark one works, and a",
        "toggle would hide the comparison.",
        "",
    ]

    for name in names:
        title = name.replace("_", " ").capitalize()
        lines += [
            f"## {title}",
            "",
            CAPTIONS[name],
            "",
            f"![{title}, light](gallery/{name}.png)",
            "",
            f"![{title}, dark](gallery/{name}-dark.png)",
            "",
        ]

    page = directory.parent / "gallery.md"
    page.write_text("\n".join(lines), encoding="utf-8")

    return page


def main() -> int:
    """Render the gallery and its page."""
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT

    names = render_all(directory)
    page = write_page(directory, names)

    print(f"Rendered {len(names) * 2} images to {directory}; wrote {page}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
