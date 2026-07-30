# tests/test_attribution.py
"""Tests for performance attribution: contributions, linking, and drags."""
import math

import numpy as np
import pandas as pd
import pytest

from beacon.analysis import (
    attribute,
    cap_drag,
    carino_factor,
    cost_drag,
    drifted_weights,
    link_contributions,
)
from beacon.exceptions import CalculationError

DATES = pd.bdate_range("2024-01-01", periods=60)
ASSETS = ["AAA", "BBB", "CCC"]


def price_frame(growth: dict[str, float],
                seed: int | None = None) -> pd.DataFrame:
    """Price paths for the constituents, optionally with real variation."""
    columns = {}
    generator = np.random.default_rng(seed) if seed is not None else None

    for name, total in growth.items():
        drift = total ** (1.0 / (len(DATES) - 1)) - 1.0
        moves = np.full(len(DATES), drift)
        if generator is not None:
            moves = moves + generator.normal(0.0, 0.005, size=len(DATES))
        moves[0] = 0.0
        columns[name] = 100.0 * (1.0 + moves).cumprod()

    return pd.DataFrame(columns, index=DATES)


class TestCarinoFactor:

    def test_limit_at_zero_is_one(self):
        assert carino_factor(0.0) == 1.0

    def test_near_zero_uses_the_limit(self):
        assert carino_factor(1e-15) == 1.0

    def test_known_value(self):
        """ln(1.5)/0.5."""
        assert carino_factor(0.5) == pytest.approx(math.log(1.5) / 0.5)

    def test_is_below_one_for_gains(self):
        assert carino_factor(0.2) < 1.0

    def test_is_above_one_for_losses(self):
        assert carino_factor(-0.2) > 1.0

    def test_total_wipeout_is_rejected(self):
        """ln(0) is undefined; substituting a number would hide that."""
        with pytest.raises(CalculationError, match="wipes out the index"):
            carino_factor(-1.0)

    def test_worse_than_wipeout_is_rejected(self):
        with pytest.raises(CalculationError):
            carino_factor(-1.5)


class TestLinking:
    """The multi-period problem: contributions must sum to a compounded total."""

    def _contributions(self,
                       seed: int = 1) -> tuple[pd.DataFrame, pd.Series]:
        generator = np.random.default_rng(seed)
        values = generator.normal(0.001, 0.01, size=(len(DATES), len(ASSETS)))
        contributions = pd.DataFrame(values, index=DATES, columns=ASSETS)

        return contributions, contributions.sum(axis=1)

    def test_linked_contributions_sum_to_the_compounded_total(self):
        contributions, period_returns = self._contributions()
        expected = float((1.0 + period_returns).prod() - 1.0)

        linked = link_contributions(contributions, period_returns)

        assert float(linked.sum()) == pytest.approx(expected, abs=1e-12)

    def test_unlinked_sum_falls_short_of_the_compounded_total(self):
        """Why linking is needed at all — the gap is not negligible."""
        contributions, period_returns = self._contributions()

        arithmetic = float(contributions.to_numpy().sum())
        compounded = float((1.0 + period_returns).prod() - 1.0)

        assert abs(compounded - arithmetic) > 1e-4

    def test_linking_preserves_relative_size(self):
        """Scaling every period by the same factor cannot reorder anything."""
        contributions, period_returns = self._contributions()

        linked = link_contributions(contributions, period_returns)
        arithmetic = contributions.sum(axis=0)

        assert list(linked.sort_values().index) == list(arithmetic.sort_values().index)

    def test_empty_contributions_give_an_empty_result(self):
        assert link_contributions(pd.DataFrame(), pd.Series(dtype=float)).empty

    def test_single_period_needs_no_adjustment(self):
        contributions = pd.DataFrame({"AAA": [0.02], "BBB": [0.01]},
                                     index=DATES[:1])
        returns = contributions.sum(axis=1)

        linked = link_contributions(contributions, returns)

        assert float(linked.sum()) == pytest.approx(0.03)


class TestDriftedWeights:

    def test_weights_start_at_the_snapshot(self):
        prices = price_frame({"AAA": 2.0, "BBB": 0.5})
        snapshots = {DATES[0]: {"AAA": 0.5, "BBB": 0.5}}

        weights = drifted_weights(snapshots, prices)

        assert weights.iloc[0]["AAA"] == pytest.approx(0.5)

    def test_weights_drift_toward_the_winner(self):
        prices = price_frame({"AAA": 2.0, "BBB": 0.5})
        snapshots = {DATES[0]: {"AAA": 0.5, "BBB": 0.5}}

        weights = drifted_weights(snapshots, prices)

        assert weights.iloc[-1]["AAA"] > weights.iloc[0]["AAA"]

    def test_weights_sum_to_one_every_day(self):
        prices = price_frame({"AAA": 2.0, "BBB": 0.5, "CCC": 1.2})
        snapshots = {DATES[0]: {"AAA": 0.4, "BBB": 0.35, "CCC": 0.25}}

        weights = drifted_weights(snapshots, prices)

        assert np.allclose(weights.sum(axis=1), 1.0)

    def test_a_later_rebalance_resets_the_weights(self):
        prices = price_frame({"AAA": 2.0, "BBB": 0.5})
        reset = DATES[30]
        snapshots = {DATES[0]: {"AAA": 0.5, "BBB": 0.5},
                     reset: {"AAA": 0.5, "BBB": 0.5}}

        weights = drifted_weights(snapshots, prices)

        assert weights.loc[reset]["AAA"] == pytest.approx(0.5)

    def test_dates_before_the_first_rebalance_are_excluded(self):
        prices = price_frame({"AAA": 2.0, "BBB": 0.5})
        snapshots = {DATES[10]: {"AAA": 0.5, "BBB": 0.5}}

        weights = drifted_weights(snapshots, prices)

        assert weights.index[0] == DATES[10]

    def test_no_snapshots_is_rejected(self):
        with pytest.raises(CalculationError, match="no weight snapshots"):
            drifted_weights({}, price_frame({"AAA": 1.1}))


class TestAttribute:

    def _setup(self,
               seed: int | None = 3):
        prices = price_frame({"AAA": 2.0, "BBB": 0.5, "CCC": 1.3}, seed=seed)
        snapshots = {DATES[0]: {"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3}}
        weights = drifted_weights(snapshots, prices)
        asset_returns = prices.pct_change()
        period_returns = (weights.shift(1) * asset_returns).sum(axis=1)

        return period_returns, weights, asset_returns

    def test_contributions_reconcile_to_the_total(self):
        """The acceptance criterion."""
        result = attribute(*self._setup())

        assert result.reconciles()
        assert result.explained == pytest.approx(result.total_return, abs=1e-12)

    def test_residual_is_machine_epsilon(self):
        result = attribute(*self._setup())

        assert abs(result.residual) < 1e-12

    def test_residual_is_reported_not_absorbed(self):
        """It must exist as its own field even when it is tiny."""
        result = attribute(*self._setup())

        assert hasattr(result, "residual")
        assert result.explained + result.residual == pytest.approx(
            result.total_return, abs=1e-15)

    def test_every_constituent_appears(self):
        result = attribute(*self._setup())

        assert {item.asset_id for item in result.contributions} == set(ASSETS)

    def test_contributions_are_ordered_largest_first(self):
        result = attribute(*self._setup())
        values = [item.contribution for item in result.contributions]

        assert values == sorted(values, reverse=True)

    def test_the_winner_contributes_positively(self):
        result = attribute(*self._setup())
        by_asset = {item.asset_id: item for item in result.contributions}

        assert by_asset["AAA"].contribution > 0
        assert by_asset["BBB"].contribution < 0

    def test_average_weight_is_reported(self):
        result = attribute(*self._setup())

        for item in result.contributions:
            assert 0.0 < item.average_weight < 1.0

    def test_asset_total_return_is_reported(self):
        result = attribute(*self._setup())
        by_asset = {item.asset_id: item for item in result.contributions}

        assert by_asset["AAA"].total_return == pytest.approx(1.0, abs=0.15)

    def test_window_and_period_count_are_reported(self):
        period_returns, weights, asset_returns = self._setup()

        result = attribute(period_returns, weights, asset_returns)

        assert result.periods == len(DATES) - 1
        assert result.start == DATES[1].isoformat()

    def test_drags_pass_through_when_supplied(self):
        period_returns, weights, asset_returns = self._setup()

        result = attribute(period_returns, weights, asset_returns,
                           cap_drag=-0.01, cost_drag=-0.002)

        assert result.cap_drag == -0.01
        assert result.cost_drag == -0.002

    def test_drags_are_none_when_not_supplied(self):
        result = attribute(*self._setup())

        assert result.cap_drag is None
        assert result.cost_drag is None

    def test_to_frame_round_trips_the_contributions(self):
        result = attribute(*self._setup())

        frame = result.to_frame()

        assert list(frame["asset_id"]) == [i.asset_id for i in result.contributions]
        assert frame["contribution"].sum() == pytest.approx(result.explained)

    def test_disjoint_inputs_are_rejected(self):
        period_returns, weights, asset_returns = self._setup()
        shifted = weights.copy()
        shifted.index = shifted.index + pd.Timedelta(days=3650)

        with pytest.raises(CalculationError, match="share no dates"):
            attribute(period_returns, shifted, asset_returns)


class TestCostDrag:

    def test_costs_reduce_the_return(self):
        assert cost_drag(1_000.0, 100_000.0) == pytest.approx(-0.01)

    def test_zero_costs_give_no_drag(self):
        assert cost_drag(0.0, 100_000.0) == 0.0

    def test_sign_is_always_negative(self):
        assert cost_drag(-500.0, 100_000.0) < 0

    def test_non_positive_capital_is_rejected(self):
        with pytest.raises(CalculationError, match="must be positive"):
            cost_drag(100.0, 0.0)


class TestCapDrag:

    def test_capping_the_winner_costs_return(self):
        """AAA doubles; holding less of it must reduce the index return."""
        prices = price_frame({"AAA": 2.0, "BBB": 0.5})
        uncapped = {DATES[0]: {"AAA": 0.8, "BBB": 0.2}}
        capped = {DATES[0]: {"AAA": 0.5, "BBB": 0.5}}

        drag = cap_drag(capped, uncapped, prices)

        assert drag < 0

    def test_capping_the_loser_would_gain(self):
        prices = price_frame({"AAA": 2.0, "BBB": 0.5})
        uncapped = {DATES[0]: {"AAA": 0.2, "BBB": 0.8}}
        capped = {DATES[0]: {"AAA": 0.5, "BBB": 0.5}}

        assert cap_drag(capped, uncapped, prices) > 0

    def test_no_cap_means_no_drag(self):
        prices = price_frame({"AAA": 2.0, "BBB": 0.5})
        weights = {DATES[0]: {"AAA": 0.5, "BBB": 0.5}}

        assert cap_drag(weights, weights, prices) == pytest.approx(0.0, abs=1e-12)


class TestEndToEndWithARealIndex:
    """Attribution of an index the calculator actually produced."""

    def _run(self,
             scheme):
        import sys

        sys.path.insert(0, "tests")
        from beacon.index.calculation import IndexCalculator
        from test_index_weighting import BASE_PRICE, END, START, build_fetcher, definition

        fetcher = build_fetcher()
        result = IndexCalculator(definition(scheme), fetcher).run(
            start_date=START, end_date=END)
        prices = pd.DataFrame({
            name: fetcher.fetch_market_data(name, START, END)["CLOSE"]
            for name in BASE_PRICE
        }).reindex(result.index_levels.index)

        return result, prices

    @pytest.mark.parametrize("scheme_name", ["EqualWeighted", "MarketCapWeighted"])
    def test_reconciles_against_the_calculated_index(self,
                                                     scheme_name):
        from beacon.index.methodology import EqualWeighted, MarketCapWeighted

        scheme = EqualWeighted() if scheme_name == "EqualWeighted" else MarketCapWeighted()
        result, prices = self._run(scheme)

        weights = drifted_weights(result.weight_snapshots, prices)
        attribution = attribute(result.index_levels.pct_change().dropna(),
                                weights, prices.pct_change())

        assert attribution.reconciles(), f"residual {attribution.residual:.3e}"

    def test_contributions_match_the_index_return(self):
        from beacon.index.methodology import EqualWeighted

        result, prices = self._run(EqualWeighted())
        weights = drifted_weights(result.weight_snapshots, prices)
        attribution = attribute(result.index_levels.pct_change().dropna(),
                                weights, prices.pct_change())

        assert attribution.explained == pytest.approx(attribution.total_return,
                                                      abs=1e-12)
