# tests/test_plot_images.py
"""BN-82: image regression over every chart, in both styles.

These are the tests that catch a change nothing else can see — a line that
moved two pixels, a label that started overlapping, a colour that shifted by a
shade. Everything in `test_plot.py` asserts a property of the axes; these
assert what the picture looks like.

## Running and updating

    pytest tests/test_plot_images.py --mpl                    # compare
    pytest tests/test_plot_images.py --mpl-generate-path=tests/baseline

Without `--mpl` these run as plain smoke tests: the figure is built and
discarded. That is deliberate — a developer without the baselines checked out,
or on a machine whose font stack differs, still gets the charts exercised
rather than a wall of failures they cannot act on. CI passes `--mpl`.

## Why the images can be stable at all

Everything comes from `beacon.testing.dataset`, whose paths avoid `exp` and
`log` so the numbers are bit-identical across platforms, and the styles pin
DejaVu Sans, which ships with matplotlib. Without both of those a baseline
would only be valid on the machine that generated it.
"""
import matplotlib
import pytest

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

END = "2024-12-31"
CAP = 0.20
RISK_FREE = 0.02

# A pixel of drift fails. That is the point: a chart nobody looked at that
# quietly moved is exactly what this catches, and a generous tolerance would
# let it through.
TOLERANCE = 0.0

MODES = [LIGHT, DARK]


@pytest.fixture(scope="module")
def charts():
    """Every chart, ready to draw. Computed once for the whole module."""
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
        data_provider=fetcher, index_result=index_result,
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

    return {
        "level": lambda: index_result.plot.level(benchmark=prices["CCC"]),
        "weights": lambda: index_result.plot.weights(),
        "performance": lambda: backtest.plot.performance(),
        "annual_returns": lambda: backtest.plot.annual_returns(),
        "contributions": lambda: attribution.plot.contributions(),
        "compare": lambda: compare(index_result, backtest),
        "exposures": lambda: optimisation.plot.exposures(),
        "frontier": lambda: optimisation.plot.frontier(
            frontier, risk_free_rate=RISK_FREE),
        "correlation": lambda: risk_model.plot.correlation(),
    }


def _draw(charts,
          name: str,
          mode: str):
    """Draw one chart in one style and return its figure."""
    plt.close("all")
    use(mode)

    axes = charts[name]()
    figure = axes.get_figure()

    use(LIGHT)

    return figure


@pytest.mark.parametrize("mode", MODES)
class TestChartImages:
    """One test per chart, run in both styles."""

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_level(self, charts, mode):
        return _draw(charts, "level", mode)

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_weights(self, charts, mode):
        return _draw(charts, "weights", mode)

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_performance(self, charts, mode):
        return _draw(charts, "performance", mode)

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_annual_returns(self, charts, mode):
        return _draw(charts, "annual_returns", mode)

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_contributions(self, charts, mode):
        return _draw(charts, "contributions", mode)

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_compare(self, charts, mode):
        return _draw(charts, "compare", mode)

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_exposures(self, charts, mode):
        return _draw(charts, "exposures", mode)

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_frontier(self, charts, mode):
        return _draw(charts, "frontier", mode)

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_correlation(self, charts, mode):
        return _draw(charts, "correlation", mode)


class TestGalleryScript:
    """The gallery is generated, so the generator is tested."""

    def _module(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "scripts" / "build_gallery.py"
        loader = importlib.util.spec_from_file_location("build_gallery", path)
        assert loader is not None and loader.loader is not None

        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)

        return module

    def test_it_renders_every_chart_in_both_styles(self,
                                                   tmp_path):
        module = self._module()

        names = module.render_all(tmp_path)
        images = sorted(path.name for path in tmp_path.glob("*.png"))

        assert len(names) == 9
        assert len(images) == 18

    def test_every_chart_has_a_caption(self):
        """A gallery of unlabelled pictures says less than a list of names."""
        module = self._module()

        assert set(module.CAPTIONS) == set(module.chart_calls(
            module.build_results()))

    def test_the_page_links_both_modes_for_each_chart(self,
                                                      tmp_path):
        module = self._module()

        names = module.render_all(tmp_path)
        page = module.write_page(tmp_path, names)
        text = page.read_text(encoding="utf-8")

        for name in names:
            assert f"gallery/{name}.png" in text
            assert f"gallery/{name}-dark.png" in text

    def test_the_gallery_covers_every_chart_the_accessors_offer(self):
        """A chart added without a gallery entry would go undocumented, and
        nobody would notice until they went looking for it."""
        module = self._module()
        from beacon.plot import accessors

        offered = set()
        for cls in (accessors.IndexPlots, accessors.BacktestPlots,
                    accessors.AttributionPlots, accessors.OptimisationPlots,
                    accessors.RiskPlots):
            offered |= {name for name in dir(cls)
                        if not name.startswith("_") and name != "methods"}

        # compare() is a free function rather than an accessor method, so it is
        # in the gallery without being in this set.
        assert offered <= set(module.CAPTIONS)
