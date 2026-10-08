# tests/test_strategy_active.py
"""BN-280: active strategies, their constructions, constraints and reports.

The universe is test_strategy_tracking's: eighteen names in three sectors
with a year of history before 2024, benchmarked against their market-cap
index. The signal is fixed per name, so which names a strategy should favour
is known in advance: SIGNAL ranks the names by their position in NAMES.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.attribution import AttributionResult
from beacon.backtest import Backtest
from beacon.expressions import data
from beacon.strategy import (
    ActiveShare,
    ActiveStrategy,
    FieldSignal,
    FunctionSignal,
    HoldingsLimit,
    MaxAlpha,
    MeanVariance,
    Momentum,
    RelativePositionBounds,
    RelativeSectorBounds,
    StrategyContext,
    TrackingErrorBudget,
    TurnoverLimit,
)
from beacon.strategy.signals import standardised
from test_strategy_tracking import DATA, END, NAMES, SECTOR_OF, START, definition

RANK = {name: position for position, name in enumerate(NAMES)}
SIGNAL = FunctionSignal(lambda name, date, fetcher: float(RANK.get(name, 0)))
CONTEXT = StrategyContext(DATA, "USD")


def strategy(**settings) -> ActiveStrategy:
    settings.setdefault("signal", SIGNAL)

    return ActiveStrategy(benchmark=definition(), **settings)


def run(active: ActiveStrategy):
    return Backtest(initial_capital=1e7, data_provider=DATA, cache=False).run(
        active, start=START, end=END)


@pytest.fixture(scope="module")
def tilted():
    return run(strategy(construction=MaxAlpha(tracking_error=0.02)))


class TestMaxAlpha:

    def test_it_spends_its_tracking_error_budget_and_no_more(self,
                                                             tilted):
        errors = [step.tracking_error for step in tilted.active]

        assert all(error <= 0.02 + 1e-6 for error in errors)
        assert all(error > 0.015 for error in errors)

    def test_it_leans_toward_the_signal(self,
                                        tilted):
        step = tilted.active[0]
        active = {name: step.weights.get(name, 0.0) - step.benchmark.get(name, 0.0)
                  for name in NAMES}

        assert active[NAMES[-1]] > 0.0
        assert active[NAMES[0]] < 0.0

    def test_a_bigger_budget_is_more_active(self,
                                            tilted):
        bold = run(strategy(construction=MaxAlpha(tracking_error=0.05)))

        assert bold.active_share().mean() > tilted.active_share().mean()

    def test_it_is_long_only_and_fully_invested(self,
                                                tilted):
        for step in tilted.active:
            assert min(step.weights.values()) >= 0.0
            assert sum(step.weights.values()) == pytest.approx(1.0)

    def test_it_rebalances_monthly(self,
                                   tilted):
        assert len(tilted.active) == 12
        assert [step.date.month for step in tilted.active] == list(range(1, 13))


class TestMeanVariance:

    def test_more_risk_aversion_holds_closer_to_the_benchmark(self):
        relaxed = run(strategy(construction=MeanVariance(risk_aversion=2.0)))
        cautious = run(strategy(construction=MeanVariance(risk_aversion=50.0)))

        assert (np.mean([step.tracking_error for step in cautious.active])
                < np.mean([step.tracking_error for step in relaxed.active]))

    def test_a_tracking_error_budget_caps_it(self):
        capped = run(strategy(construction=MeanVariance(risk_aversion=1.0),
                              constraints=[TrackingErrorBudget(0.01)]))

        assert all(step.tracking_error <= 0.01 + 1e-6 for step in capped.active)


class TestConstraints:

    def test_active_share_has_a_floor(self):
        result = run(strategy(construction=MeanVariance(risk_aversion=200.0),
                              constraints=[ActiveShare(minimum=0.25)]))

        assert all(step.active_share >= 0.25 - 1e-6 for step in result.active)

    def test_sectors_stay_near_the_benchmark(self):
        result = run(strategy(constraints=[RelativeSectorBounds(within=0.02)]))

        for step in result.active:
            for sector in ("Tech", "Health", "Energy"):
                held = sum(w for n, w in step.weights.items() if SECTOR_OF[n] == sector)
                wanted = sum(w for n, w in step.benchmark.items() if SECTOR_OF[n] == sector)

                assert abs(held - wanted) <= 0.02 + 1e-6

    def test_positions_stay_near_the_benchmark(self):
        result = run(strategy(constraints=[RelativePositionBounds(within=0.01)]))

        for step in result.active:
            assert all(abs(step.weights.get(name, 0.0) - step.benchmark.get(name, 0.0))
                       <= 0.01 + 1e-6 for name in NAMES)

    def test_a_holdings_limit(self):
        result = run(strategy(constraints=[HoldingsLimit(8)]))

        assert all(len(step.weights) <= 8 for step in result.active)

    def test_turnover_is_limited_after_the_first_rebalance(self):
        result = run(strategy(constraints=[TurnoverLimit(0.05)]))

        for earlier, later in zip(result.active, result.active[1:], strict=False):
            names = set(earlier.weights) | set(later.weights)
            traded = sum(abs(later.weights.get(n, 0.0) - earlier.weights.get(n, 0.0))
                         for n in names) / 2.0

            assert traded <= 0.05 + 1e-6


class TestCandidates:

    def test_a_screened_out_name_is_not_held(self):
        screen = data.reference.sector != "Energy"
        result = run(strategy(screen=screen))

        for step in result.active:
            assert not any(SECTOR_OF[name] == "Energy" for name in step.weights)

    def test_a_universe_limits_what_is_held(self):
        chosen = NAMES[:9]
        result = run(strategy(universe=chosen))

        assert all(set(step.weights) <= set(chosen) for step in result.active)


class TestTheResult:

    def test_the_summary_reports_on_the_strategy(self,
                                                 tilted):
        summary = tilted.summary()

        assert {"information_ratio", "average_active_share",
                "average_ex_ante_tracking_error"} <= set(summary)
        assert summary["information_ratio"] is not None

    def test_the_run_is_measured_against_the_benchmark(self,
                                                       tilted):
        assert tilted.index.target is not None
        assert tilted.get_tracking_error() > 0.0

    def test_active_attribution_explains_the_active_return(self,
                                                           tilted):
        attribution = tilted.active_attribution()

        assert isinstance(attribution, AttributionResult)
        assert attribution.contributions

    def test_a_passive_run_has_no_active_record(self):
        result = Backtest(initial_capital=1e7, data_provider=DATA, cache=False).run(
            definition(), start=START, end=END)

        assert result.active == []
        assert "information_ratio" not in result.summary()


class TestSignals:

    def test_scores_are_standardised_and_capped(self):
        scores = standardised({"a": 1.0, "b": 2.0, "c": 3.0, "d": None})

        assert scores["d"] == 0.0
        assert scores["b"] == pytest.approx(0.0)
        assert scores["a"] == pytest.approx(-scores["c"])

        extreme = standardised({**{f"n{i}": 0.0 for i in range(99)}, "x": 1e6})

        assert extreme["x"] == 3.0

    def test_a_flat_signal_scores_nothing(self):
        assert set(standardised({"a": 1.0, "b": 1.0}).values()) == {0.0}

    def test_lower_can_be_better(self):
        lower = FunctionSignal(lambda name, date, fetcher: float(RANK[name]),
                               higher_is_better=False)
        scores = lower.scores(NAMES, pd.Timestamp(START), CONTEXT)

        assert scores[NAMES[0]] > scores[NAMES[-1]]

    def test_momentum_is_the_return_without_the_last_month(self):
        momentum = Momentum(lookback_days=126, skip_days=21)
        values = momentum.values(NAMES[:2], pd.Timestamp(START), CONTEXT)
        prices = DATA.fetch_prices(NAMES[:2], "2023-01-03", START)
        window = prices.ffill().pct_change().iloc[1:].tail(126).iloc[:-21]

        assert values[NAMES[0]] == pytest.approx(
            float((1 + window[NAMES[0]]).prod() - 1))

    def test_a_field_signal_reads_the_field(self):
        values = FieldSignal(data.market.close).values(NAMES[:1], pd.Timestamp(START),
                                                       CONTEXT)

        assert values[NAMES[0]] > 0.0


class TestTheSettingsAreChecked:

    @pytest.mark.parametrize("build", [
        lambda: MaxAlpha(tracking_error=0.0),
        lambda: MeanVariance(risk_aversion=0.0),
        ActiveShare,
        lambda: ActiveShare(minimum=0.6, maximum=0.4),
        lambda: RelativeSectorBounds(within=-0.1),
        lambda: HoldingsLimit(0),
        lambda: Momentum(lookback_days=20, skip_days=21),
        lambda: strategy(rebalancing="HOURLY"),
    ])
    def test_an_impossible_setting_is_refused(self,
                                              build):
        with pytest.raises(ValueError):
            build()
