# tests/test_plot.py
"""BN-77 to BN-81: the plot accessors, the beacon styles and every chart.

Rendering is checked by interrogating the axes rather than by comparing images.
Image regression is BN-82's job and needs baselines; these tests assert the
things a baseline cannot — that a line has the accent colour, that an annotated
total matches the bars beside it, that a capital market line is tangent — which
is what makes a baseline worth trusting once it exists.
"""
import subprocess
import sys

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
from beacon.plot import compare, style, use
from beacon.risk import estimate_risk_model
from beacon.testing import dataset
from beacon.tokens import DARK, LIGHT, colour

END = "2024-12-31"
# Market-cap weights are uneven, so a cap at this level genuinely binds and
# the weights chart has a marker to draw. Equal weighting hands every name 1/n,
# where no cap above that can ever bite.
CAP = 0.20
RISK_FREE = 0.02


@pytest.fixture(autouse=True)
def _clean_figures():
    """Every test starts and ends with no open figures."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def index_result():
    definition = IndexDefinition(
        index_id="CANON", index_name="Canonical", base_date=dataset.START,
        base_value=1000.0, currency="USD", eligibility_rules=[],
        weighting_scheme=MarketCapWeighted(), rebalancing_frequency="QUARTERLY",
        universe_identifiers=list(dataset.UNIVERSE),
        max_constituent_weight=CAP)

    return IndexCalculator(definition, dataset.data_fetcher()).run(
        start_date=dataset.START, end_date=END)


@pytest.fixture(scope="module")
def backtest(index_result):
    return BacktestEngine(start_date=dataset.START, end_date=END,
                          initial_capital=10_000_000.0,
                          data_provider=dataset.data_fetcher(),
                          target_index_result=index_result,
                          transaction_cost_bps=10.0).run()


@pytest.fixture(scope="module")
def attribution(index_result):
    prices = dataset.prices().loc[:END]
    weights = drifted_weights(index_result.weight_snapshots, prices)
    asset_returns = prices.pct_change().reindex(weights.index)
    period_returns = (weights.shift(1) * asset_returns).sum(axis=1)

    return attribute(period_returns, weights, asset_returns,
                     cap_drag=-0.003, cost_drag=-0.004)


@pytest.fixture(scope="module")
def risk_model():
    return estimate_risk_model(dataset.returns(), intensity=0.1)


@pytest.fixture(scope="module")
def optimisation(risk_model):
    return minimise_tracking_error(
        dataset.equal_weights(),
        [FullInvestment(), PositionBounds(0.0, 0.25),
         GroupBounds("Technology", dataset.sectors()["Technology"],
                     maximum=CAP)],
        risk_model)


@pytest.fixture(scope="module")
def frontier(risk_model):
    returns = {name: 0.04 + 0.02 * position
               for position, name in enumerate(dataset.UNIVERSE)}

    return efficient_frontier(risk_model, returns, points=12,
                              risk_free_rate=RISK_FREE)


class TestLazyAccessor:
    """The acceptance criterion: importing beacon costs nothing extra."""

    def test_importing_beacon_does_not_import_matplotlib(self):
        """A user who never draws a chart never pays for the ability to."""
        script = ("import sys\n"
                  "import beacon\n"
                  "from beacon.index.result import IndexResult\n"
                  "from beacon.risk.model import RiskModel\n"
                  "assert 'matplotlib' not in sys.modules, sorted(\n"
                  "    m for m in sys.modules if 'matplotlib' in m)\n"
                  "print('ok')\n")

        completed = subprocess.run([sys.executable, "-c", script],
                                   capture_output=True, text=True, check=False)

        assert completed.returncode == 0, completed.stderr
        assert "ok" in completed.stdout

    def test_the_plot_package_imports_without_matplotlib(self):
        """The descriptor lives in a module that needs nothing heavy."""
        script = ("import sys\n"
                  "class Blocker:\n"
                  "    def find_spec(self, name, path=None, target=None):\n"
                  "        if name.split('.')[0] == 'matplotlib':\n"
                  "            raise ImportError(name)\n"
                  "        return None\n"
                  "sys.meta_path.insert(0, Blocker())\n"
                  "import beacon.plot\n"
                  "from beacon.plot.base import PlotAccessor\n"
                  "print('ok')\n")

        completed = subprocess.run([sys.executable, "-c", script],
                                   capture_output=True, text=True, check=False)

        assert completed.returncode == 0, completed.stderr

    def test_touching_the_accessor_imports_matplotlib(self,
                                                      index_result):
        """The other half: it does resolve when asked."""
        assert index_result.plot is not None

    def test_the_accessor_lists_its_charts(self,
                                           index_result):
        """A caller who types `result.plot` should be told what they can do."""
        assert repr(index_result.plot) == "<IndexPlots: level(), weights()>"

    def test_each_result_gets_its_own_accessor(self,
                                               index_result,
                                               backtest,
                                               risk_model,
                                               optimisation,
                                               attribution):
        names = {type(result.plot).__name__
                 for result in (index_result, backtest, risk_model,
                                optimisation, attribution)}

        assert names == {"IndexPlots", "BacktestPlots", "RiskPlots",
                         "OptimisationPlots", "AttributionPlots"}

    def test_accessing_it_on_the_class_returns_the_descriptor(self):
        """So help() and hasattr do not import matplotlib to answer."""
        from beacon.index.result import IndexResult
        from beacon.plot.base import PlotAccessor

        assert isinstance(IndexResult.plot, PlotAccessor)


class TestStyle:
    """The acceptance criterion: `plt.style.use("beacon")` restyles a plot."""

    def test_both_styles_register(self):
        style.register()

        assert style.BEACON in plt.style.available
        assert style.BEACON_DARK in plt.style.available

    def test_it_restyles_a_plot_this_library_never_drew(self):
        style.register()

        with plt.style.context(style.BEACON):
            figure, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            from matplotlib.colors import to_hex

            assert to_hex(figure.get_facecolor()) == colour("canvas", LIGHT)
            assert to_hex(ax.get_lines()[0].get_color()) == colour("accent", LIGHT)

    def test_the_dark_style_uses_the_dark_tokens(self):
        style.register()

        with plt.style.context(style.BEACON_DARK):
            figure, _ = plt.subplots()

            from matplotlib.colors import to_hex

            assert to_hex(figure.get_facecolor()) == colour("canvas", DARK)

    def test_the_colours_come_from_the_tokens(self):
        """Not retyped: a palette written here would be right on the day it was
        written."""
        for mode in (LIGHT, DARK):
            assert style.palette(mode)["accent"] == colour("accent", mode)

    def test_gridlines_are_on_the_y_axis_and_behind_the_data(self):
        rc = style.style_dict(LIGHT)

        assert rc["axes.grid"] is True
        assert rc["axes.grid.axis"] == "y"
        assert rc["axes.axisbelow"] is True

    def test_the_top_and_right_spines_are_off(self):
        rc = style.style_dict(LIGHT)

        assert rc["axes.spines.top"] is False
        assert rc["axes.spines.right"] is False

    def test_the_series_cycle_avoids_the_signal_colours(self):
        """Green and red mean up and down elsewhere, so a third series that
        happened to be green would read as a gain."""
        ink = style.palette(LIGHT)
        cycle = style.style_dict(LIGHT)["axes.prop_cycle"].by_key()["color"]

        assert ink["success"] not in cycle
        assert ink["danger"] not in cycle

    def test_the_font_is_pinned(self):
        """Falling through to whatever the system has would make image
        regression meaningless."""
        assert style.style_dict(LIGHT)["font.sans-serif"] == ["DejaVu Sans"]

    def test_registering_twice_is_harmless(self):
        style.register()
        style.register()

        assert style.BEACON in plt.style.available

    def test_use_applies_a_style_globally(self):
        use(DARK)

        from matplotlib.colors import to_hex

        assert to_hex(plt.rcParams["figure.facecolor"]) == colour("canvas", DARK)

        use(LIGHT)


class TestCoreCharts:
    """BN-78."""

    def test_level_draws_the_index(self,
                                   index_result):
        ax = index_result.plot.level()

        assert len(ax.get_lines()) >= 1
        assert ax.get_title(loc="left") == "Index level"

    def test_level_rebases_to_one_hundred(self,
                                          index_result):
        ax = index_result.plot.level()
        first = ax.get_lines()[0].get_ydata()[0]

        assert first == pytest.approx(100.0, abs=1e-9)

    def test_level_uses_the_accent_colour(self,
                                          index_result):
        from matplotlib.colors import to_hex

        use(LIGHT)
        ax = index_result.plot.level()

        assert to_hex(ax.get_lines()[0].get_color()) == colour("accent", LIGHT)

    def test_level_draws_a_benchmark_subordinate(self,
                                                 index_result):
        """The comparison should read as secondary without being hidden."""
        from matplotlib.colors import to_hex

        use(LIGHT)
        benchmark = dataset.prices()["CCC"].loc[:END]
        ax = index_result.plot.level(benchmark=benchmark)

        lines = ax.get_lines()
        benchmark_line = next(line for line in lines
                              if to_hex(line.get_color())
                              == colour("text-secondary", LIGHT))

        assert benchmark_line.get_linewidth() < lines[0].get_linewidth()

    def test_level_marks_the_last_value(self,
                                        index_result):
        ax = index_result.plot.level()

        assert ax.texts, "expected a last-value annotation"

    def test_weights_draws_a_bar_per_constituent(self,
                                                 index_result):
        ax = index_result.plot.weights()

        assert len(ax.patches) == len(dataset.UNIVERSE)

    def test_weights_marks_the_cap(self,
                                   index_result):
        ax = index_result.plot.weights()
        vertical = [line for line in ax.get_lines()
                    if line.get_linestyle() == "--"]

        assert vertical, "expected a cap marker"
        assert vertical[0].get_xdata()[0] == pytest.approx(CAP)

    def test_weights_at_a_chosen_rebalance(self,
                                           index_result):
        first = min(index_result.weight_snapshots)
        ax = index_result.plot.weights(date=first)

        assert str(first.date()) in ax.texts[-1].get_text()

    def test_performance_draws_two_linked_panels(self,
                                                 backtest):
        ax = backtest.plot.performance()
        figure = ax.get_figure()

        assert len(figure.get_axes()) == 2
        assert figure.get_axes()[1].get_ylabel() == "Drawdown"

    def test_the_drawdown_panel_is_never_positive(self,
                                                  backtest):
        ax = backtest.plot.performance()
        lower = ax.get_figure().get_axes()[1]
        values = lower.get_lines()[0].get_ydata()

        assert max(values) <= 1e-12

    def test_the_panels_share_an_x_axis(self,
                                        backtest):
        """A drawdown is only meaningful against the path that produced it."""
        ax = backtest.plot.performance()
        upper, lower = ax.get_figure().get_axes()

        assert upper.get_xlim() == lower.get_xlim()

    def test_annual_returns_are_signed_bars(self,
                                            backtest):
        from matplotlib.colors import to_hex

        use(LIGHT)
        ax = backtest.plot.annual_returns()
        colours = {to_hex(patch.get_facecolor()) for patch in ax.patches}

        assert colours <= {colour("success", LIGHT), colour("danger", LIGHT)}

    def test_annual_returns_has_a_bar_per_year(self,
                                               backtest):
        ax = backtest.plot.annual_returns()

        assert len(ax.patches) == 2  # 2023 and 2024


class TestAttributionChart:
    """BN-79."""

    def test_it_draws_a_bar_per_constituent(self,
                                            attribution):
        ax = attribution.plot.contributions()

        assert len(ax.patches) == len(dataset.UNIVERSE)

    def test_the_annotated_total_matches_the_bars(self,
                                                  attribution):
        """The acceptance criterion, and the reason the renderer recomputes it:
        an annotation claiming a total that does not match the bars beside it
        is worse than no annotation."""
        ax = attribution.plot.contributions()
        drawn = sum(patch.get_width() for patch in ax.patches)

        note = ax.texts[-1].get_text()
        stated = float(note.split("sum to ")[1].split("%")[0]) / 100.0

        assert stated == pytest.approx(drawn, abs=5e-5)

    def test_the_total_matches_the_result(self,
                                          attribution):
        ax = attribution.plot.contributions()
        drawn = sum(patch.get_width() for patch in ax.patches)

        assert drawn == pytest.approx(attribution.total_return, abs=1e-9)

    def test_the_drags_are_reported_as_their_own_rows(self,
                                                      attribution):
        """Comparisons against a counterfactual, not terms in the
        decomposition."""
        note = attribution.plot.contributions().texts[-1].get_text()

        assert "Cap drag" in note
        assert "Cost drag" in note

    def test_bars_are_sign_coloured(self,
                                    attribution):
        from matplotlib.colors import to_hex

        use(LIGHT)
        ax = attribution.plot.contributions()

        for patch in ax.patches:
            expected = colour("success" if patch.get_width() >= 0 else "danger",
                              LIGHT)
            assert to_hex(patch.get_facecolor()) == expected


class TestCompare:
    """BN-79."""

    def test_it_draws_a_line_per_result(self,
                                        index_result,
                                        backtest):
        ax = compare(index_result, backtest)

        assert len(ax.get_lines()) == 2

    def test_every_line_starts_at_one_hundred(self,
                                              index_result,
                                              backtest):
        """The acceptance criterion: rebased on the common window, so the
        comparison is of shape rather than of span."""
        ax = compare(index_result, backtest)

        for line in ax.get_lines():
            assert line.get_ydata()[0] == pytest.approx(100.0, abs=1e-9)

    def test_the_lines_cover_the_same_dates(self,
                                            index_result,
                                            backtest):
        ax = compare(index_result, backtest)
        first, second = (line.get_xdata() for line in ax.get_lines())

        assert len(first) == len(second)
        assert first[0] == second[0]
        assert first[-1] == second[-1]

    def test_it_annotates_the_metrics(self,
                                      index_result,
                                      backtest):
        ax = compare(index_result, backtest)
        note = ax.texts[-1].get_text()

        assert "Total return" in note
        assert "shared observations" in note

    def test_labels_can_be_supplied(self,
                                    index_result,
                                    backtest):
        ax = compare(index_result, backtest, labels=["Index", "Portfolio"])

        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "Index", "Portfolio"]

    def test_one_result_is_refused(self,
                                   index_result):
        with pytest.raises(ValueError, match="at least two"):
            compare(index_result)

    def test_results_that_never_overlap_are_refused(self,
                                                    index_result):
        import copy

        import pandas as pd

        elsewhere = copy.copy(index_result)
        elsewhere.index_levels = pd.Series(
            [1.0, 2.0], index=pd.to_datetime(["2015-01-01", "2015-01-02"]))

        with pytest.raises(ValueError, match="share no dates"):
            compare(index_result, elsewhere)

    def test_something_without_a_level_is_refused(self,
                                                  index_result,
                                                  risk_model):
        with pytest.raises(TypeError, match="carries no level series"):
            compare(index_result, risk_model)


class TestOptimiserCharts:
    """BN-80."""

    def test_exposures_draws_a_tilt_per_name(self,
                                             optimisation):
        ax = optimisation.plot.exposures()

        assert len(ax.patches) == len(dataset.UNIVERSE)

    def test_tilts_are_sign_coloured(self,
                                     optimisation):
        from matplotlib.colors import to_hex

        use(LIGHT)
        ax = optimisation.plot.exposures()

        for patch in ax.patches:
            expected = colour("success" if patch.get_width() >= 0 else "danger",
                              LIGHT)
            assert to_hex(patch.get_facecolor()) == expected

    def test_the_tilts_sum_to_zero(self,
                                   optimisation):
        """Both sides fully invested, so rearranging cannot create weight."""
        ax = optimisation.plot.exposures()

        assert sum(patch.get_width() for patch in ax.patches) == pytest.approx(
            0.0, abs=1e-9)

    def test_frontier_draws_the_curve_and_the_named_points(self,
                                                           optimisation,
                                                           frontier):
        ax = optimisation.plot.frontier(frontier, risk_free_rate=RISK_FREE)
        labels = {text.get_text() for text in ax.get_legend().get_texts()}

        assert {"Frontier", "Capital market line", "Minimum variance",
                "Tangency"} <= labels

    def test_the_capital_market_line_is_tangent(self,
                                                optimisation,
                                                frontier):
        """It has to pass through the tangency point — that is what tangency
        means, and an inline excess-return expression once made it miss.
        `x or 0.0 - rf` parses as `x or (0.0 - rf)`, which silently drops the
        risk-free rate and leaves a plausible line tangent to nothing.
        """
        ax = optimisation.plot.frontier(frontier, risk_free_rate=RISK_FREE)
        line = next(entry for entry in ax.get_lines()
                    if entry.get_linestyle() == "--")

        x, y = line.get_xdata(), line.get_ydata()
        slope = (y[1] - y[0]) / (x[1] - x[0])
        at_tangency = y[0] + slope * frontier.tangency.volatility

        assert at_tangency == pytest.approx(frontier.tangency.expected_return,
                                            abs=1e-12)

    def test_the_capital_market_line_starts_at_the_risk_free_rate(self,
                                                                   optimisation,
                                                                   frontier):
        ax = optimisation.plot.frontier(frontier, risk_free_rate=RISK_FREE)
        line = next(entry for entry in ax.get_lines()
                    if entry.get_linestyle() == "--")

        assert line.get_ydata()[0] == pytest.approx(RISK_FREE)

    def test_the_tangency_sharpe_is_annotated(self,
                                              optimisation,
                                              frontier):
        ax = optimisation.plot.frontier(frontier, risk_free_rate=RISK_FREE)
        annotations = " ".join(text.get_text() for text in ax.texts)

        assert "Sharpe" in annotations
        assert f"{frontier.tangency.sharpe_ratio:.2f}" in annotations


class TestCorrelationChart:
    """BN-81."""

    def test_the_colormap_is_importable_standalone(self):
        """The acceptance criterion: usable without drawing a Beacon chart."""
        from matplotlib import colormaps

        style.register()

        assert style.CORRELATION_COLORMAP in colormaps
        assert colormaps[style.CORRELATION_COLORMAP](0.5) is not None

    def test_the_colormap_runs_green_to_red(self):
        from matplotlib.colors import to_hex

        from beacon.tokens import raw_colours

        colormap = style.correlation_colormap()
        raw = raw_colours()

        assert to_hex(colormap(0.0)) == raw["heatmap-low"]
        assert to_hex(colormap(1.0)) == raw["heatmap-high"]

    def test_the_heatmap_is_symmetric(self,
                                      risk_model):
        """The acceptance criterion, checked on the array that was drawn."""
        import numpy as np

        ax = risk_model.plot.correlation()
        values = ax.images[0].get_array()

        assert np.allclose(values, values.T, atol=1e-12)

    def test_the_diagonal_is_at_the_top_of_the_scale(self,
                                                     risk_model):
        """Unit correlation, so it renders in the hottest colour — the red
        diagonal the acceptance criterion asks for."""
        import numpy as np

        ax = risk_model.plot.correlation()
        values = np.asarray(ax.images[0].get_array())

        assert np.allclose(np.diag(values), 1.0, atol=1e-12)
        assert ax.images[0].get_clim()[1] == pytest.approx(1.0)

    def test_the_scale_starts_above_zero(self,
                                         risk_model):
        """Below 0.2 the difference between 0.05 and 0.1 is noise on any real
        estimate, and giving it colour implies a precision it does not have."""
        ax = risk_model.plot.correlation()

        assert ax.images[0].get_clim()[0] == pytest.approx(
            style.CORRELATION_DOMAIN[0])

    def test_it_is_labelled_by_asset(self,
                                     risk_model):
        ax = risk_model.plot.correlation()
        labels = [text.get_text() for text in ax.get_yticklabels()]

        assert labels == list(dataset.UNIVERSE)

    def test_the_colorbar_says_what_the_colours_mean(self,
                                                     risk_model):
        ax = risk_model.plot.correlation()
        bar_axes = [axis for axis in ax.get_figure().get_axes() if axis is not ax]

        assert bar_axes
        assert "correlated" in bar_axes[0].get_ylabel()


class TestComposition:
    """Every chart honours `ax=`, so it composes into a caller's figure."""

    @pytest.mark.parametrize("kind", ["level", "weights", "annual_returns",
                                      "contributions", "exposures",
                                      "correlation"])
    def test_a_supplied_axes_is_drawn_on(self,
                                         kind,
                                         index_result,
                                         backtest,
                                         attribution,
                                         optimisation,
                                         risk_model):
        figure, axes = plt.subplots(1, 2)

        call = {
            "level": lambda ax: index_result.plot.level(ax=ax),
            "weights": lambda ax: index_result.plot.weights(ax=ax),
            "annual_returns": lambda ax: backtest.plot.annual_returns(ax=ax),
            "contributions": lambda ax: attribution.plot.contributions(ax=ax),
            "exposures": lambda ax: optimisation.plot.exposures(ax=ax),
            "correlation": lambda ax: risk_model.plot.correlation(ax=ax),
        }[kind]

        returned = call(axes[1])

        assert returned is axes[1]
        assert returned.get_figure() is figure

    def test_performance_says_it_ignores_a_supplied_axes(self,
                                                         backtest,
                                                         caplog):
        """It owns a two-panel figure. Silently ignoring the argument would be
        worse than saying so."""
        _, ax = plt.subplots()

        with caplog.at_level("WARNING"):
            returned = backtest.plot.performance(ax=ax)

        assert returned is not ax
        assert "ignored" in caplog.text


class TestBothStyles:
    """Every chart renders in light and dark."""

    @pytest.mark.parametrize("mode", [LIGHT, DARK])
    def test_every_chart_renders(self,
                                 mode,
                                 index_result,
                                 backtest,
                                 attribution,
                                 optimisation,
                                 frontier,
                                 risk_model):
        use(mode)

        charts = [
            lambda: index_result.plot.level(),
            lambda: index_result.plot.weights(),
            lambda: backtest.plot.performance(),
            lambda: backtest.plot.annual_returns(),
            lambda: attribution.plot.contributions(),
            lambda: optimisation.plot.exposures(),
            lambda: optimisation.plot.frontier(frontier,
                                               risk_free_rate=RISK_FREE),
            lambda: risk_model.plot.correlation(),
            lambda: compare(index_result, backtest),
        ]

        for chart in charts:
            assert chart() is not None
            plt.close("all")

        use(LIGHT)

    def test_charts_pick_up_the_mode_from_their_own_figure(self,
                                                           index_result):
        """Read off the axes rather than from global state, so a caller can
        style one figure without affecting another."""
        from matplotlib.colors import to_hex

        use(LIGHT)

        with plt.style.context(style.BEACON_DARK):
            ax = index_result.plot.level()

            assert to_hex(ax.get_lines()[0].get_color()) == colour("accent", DARK)


class TestTruncation:
    """A chart with more names than it can label says so."""

    def test_a_wide_universe_is_truncated_with_a_note(self,
                                                      optimisation):
        """Silently dropping the tail would make a chart of forty names look
        like a chart of twenty-five."""
        import pandas as pd

        from beacon.plot.accessors import MAX_BARS

        wide = optimisation
        names = [f"A{index:02d}" for index in range(MAX_BARS + 10)]
        wide.weights = pd.Series([1.0 / len(names)] * len(names), index=names)
        wide.target_weights = pd.Series(
            [1.0 / len(names) + (0.001 * index) for index in range(len(names))],
            index=names)

        ax = wide.plot.exposures()

        assert len(ax.patches) == MAX_BARS
        assert "10 smaller tilt(s) not shown" in ax.texts[-1].get_text()

    def test_truncation_keeps_the_largest_by_magnitude(self):
        """The names worth seeing are the extremes, in either direction."""
        from beacon.plot.accessors import _truncate

        labels, _, dropped = _truncate(
            ["big-up", "tiny", "big-down"], [0.5, 0.001, -0.6], limit=2)

        assert set(labels) == {"big-up", "big-down"}
        assert dropped == 1

    def test_a_short_list_is_untouched(self):
        from beacon.plot.accessors import _truncate

        labels, values, dropped = _truncate(["a", "b"], [1.0, 2.0], limit=5)

        assert (labels, values, dropped) == (["a", "b"], [1.0, 2.0], 0)

    def test_a_truncated_weights_chart_says_so(self,
                                               index_result):
        import copy

        from beacon.plot.accessors import MAX_BARS

        crowded = copy.copy(index_result)
        latest = max(index_result.weight_snapshots)
        count = MAX_BARS + 4
        crowded.weight_snapshots = {
            latest: {f"N{index:02d}": 1.0 / count for index in range(count)}}
        crowded.cap_reports = {}

        ax = crowded.plot.weights()

        assert len(ax.patches) == MAX_BARS
        assert "4 smaller holding(s) not shown" in ax.texts[-1].get_text()


class TestDegenerateSeries:

    def test_an_empty_series_draws_no_marker(self):
        """Nothing to mark, and indexing an empty series would raise."""
        import pandas as pd

        from beacon.plot.accessors import _mark_last

        _, ax = plt.subplots()
        _mark_last(ax, pd.Series(dtype=float))

        assert len(ax.texts) == 0

    def test_a_result_without_an_identifier_is_labelled_by_position(self):
        """So a comparison of anonymous results still has a legend."""
        from beacon.plot.comparison import _label_of

        class Nameless:
            pass

        assert _label_of(Nameless(), 0) == "Series 1"
