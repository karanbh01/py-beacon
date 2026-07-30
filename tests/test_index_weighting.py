# tests/test_index_weighting.py
"""BN-103: the weighting scheme must actually drive the index level."""
import numpy as np
import pandas as pd
import pytest

from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted, MarketCapWeighted

START = "2024-01-01"
END = "2024-06-28"
DATES = pd.bdate_range(START, END)

# Equal shares, but very different prices: market-cap weighting leans 2:1
# toward AAA while equal weighting is 1:1. AAA doubles, BBB halves, so the two
# schemes must give visibly different answers.
BASE_PRICE = {"AAA": 100.0, "BBB": 50.0}
GROWTH = {"AAA": 2.0, "BBB": 0.5}
SHARES = 1_000


def build_fetcher(shares: dict[str, float] | None = None) -> DataFetcher:
    """Two names on deterministic opposing geometric paths."""
    span = len(DATES) - 1
    counts = shares or dict.fromkeys(BASE_PRICE, SHARES)
    rows = [
        {"IDENTIFIER": name,
         "DATE": date,
         "CLOSE": BASE_PRICE[name] * (GROWTH[name] ** (index / span)),
         "VOLUME": 1_000_000,
         "SHARES_OUTSTANDING": counts[name]}
        for name in BASE_PRICE
        for index, date in enumerate(DATES)
    ]
    reference = pd.DataFrame([
        {"IDENTIFIER": name, "DATE_FROM": "2020-01-01", "NAME": name,
         "CURRENCY": "USD", "EXCHANGE": "NYSE"}
        for name in BASE_PRICE
    ])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def definition(scheme,
               frequency: str = "ANNUAL") -> IndexDefinition:
    """An index over both names. ANNUAL keeps the window rebalance-free."""
    return IndexDefinition(index_id="W",
                           index_name="Weighting test",
                           base_date=START,
                           base_value=1000.0,
                           currency="USD",
                           eligibility_rules=[],
                           weighting_scheme=scheme,
                           rebalancing_frequency=frequency,
                           universe_identifiers=list(BASE_PRICE))


def run(scheme,
        frequency: str = "ANNUAL",
        fetcher: DataFetcher | None = None) -> pd.Series:
    """Run the index and return its level series."""
    calculator = IndexCalculator(definition(scheme, frequency),
                                 fetcher or build_fetcher())

    return calculator.run(start_date=START, end_date=END).index_levels


class TestTheSchemeChangesTheLevel:
    """The defect: both schemes produced byte-identical levels."""

    def test_equal_and_market_cap_differ(self):
        equal = run(EqualWeighted())
        market_cap = run(MarketCapWeighted())

        assert (equal - market_cap).abs().max() > 1.0

    def test_equal_weighted_return_is_the_average_of_the_two(self):
        """Two names, one doubling and one halving, no rebalance in the window.

        An equal-weighted buy-and-hold ends at 0.5 x 2.0 + 0.5 x 0.5 = 1.25.
        """
        levels = run(EqualWeighted())

        assert levels.iloc[-1] / levels.iloc[0] - 1 == pytest.approx(0.25, abs=1e-9)

    def test_market_cap_weighted_return_is_cap_weighted(self):
        """AAA starts at 2/3 of market cap, BBB at 1/3.

        2/3 x 2.0 + 1/3 x 0.5 = 1.5.
        """
        levels = run(MarketCapWeighted())

        assert levels.iloc[-1] / levels.iloc[0] - 1 == pytest.approx(0.50, abs=1e-9)

    def test_equal_weighting_trails_when_the_big_name_wins(self):
        """AAA is both larger and the winner, so cap weighting must lead."""
        equal = run(EqualWeighted())
        market_cap = run(MarketCapWeighted())

        assert equal.iloc[-1] < market_cap.iloc[-1]


class TestMarketCapIsUnchanged:
    """Market-cap weighting was correct by accident; it must stay correct."""

    def test_level_is_total_market_value_over_the_divisor(self):
        fetcher = build_fetcher()
        calculator = IndexCalculator(definition(MarketCapWeighted()), fetcher)
        result = calculator.run(start_date=START, end_date=END)

        date = result.index_levels.index[len(DATES) // 2]
        expected_market_value = sum(
            float(fetcher.fetch_market_data(name, str(date.date()),
                                            str(date.date()))["CLOSE"].iloc[0]) * SHARES
            for name in BASE_PRICE)

        divisor = result.divisor_history.loc[date]

        assert result.index_levels.loc[date] == pytest.approx(
            expected_market_value / divisor)

    def test_starts_at_the_base_value(self):
        assert run(MarketCapWeighted()).iloc[0] == pytest.approx(1000.0)

    def test_equal_weighted_also_starts_at_the_base_value(self):
        assert run(EqualWeighted()).iloc[0] == pytest.approx(1000.0)


class TestWeightsDriftBetweenRebalances:
    """Units are fixed between rebalances, so weights move with performance."""

    def test_an_equal_weighted_index_is_not_rebalanced_daily(self):
        """Daily rebalancing would give the arithmetic mean of daily returns.

        Holding fixed units gives the buy-and-hold result instead, which for
        a doubling and a halving is 1.25 rather than something higher.
        """
        levels = run(EqualWeighted())
        total = levels.iloc[-1] / levels.iloc[0]

        assert total == pytest.approx(1.25, abs=1e-9)

    def test_rebalancing_changes_the_outcome(self):
        """A rebalanced equal-weight index differs from buy-and-hold."""
        held = run(EqualWeighted(), frequency="ANNUAL")
        rebalanced = run(EqualWeighted(), frequency="MONTHLY")

        assert held.iloc[-1] != pytest.approx(rebalanced.iloc[-1], abs=1e-6)


class TestContributionIdentity:
    """return_t = sum_i w_{i,t-1} * r_{i,t} with the methodology's weights."""

    def _weights_and_returns(self,
                             scheme) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        fetcher = build_fetcher()
        calculator = IndexCalculator(definition(scheme), fetcher)
        result = calculator.run(start_date=START, end_date=END)

        prices = pd.DataFrame({
            name: fetcher.fetch_market_data(name, START, END)["CLOSE"]
            for name in BASE_PRICE
        }).reindex(result.index_levels.index)

        # Reconstruct the held units from the base-date weights, which is what
        # the index holds for the whole window at ANNUAL frequency.
        snapshot = result.weight_snapshots[min(result.weight_snapshots)]
        first = prices.iloc[0]
        aggregate = float((first * SHARES).sum())
        units = {name: snapshot[name] * aggregate / first[name] for name in snapshot}

        values = prices * pd.Series(units)
        weights = values.div(values.sum(axis=1), axis=0)

        return weights, prices.pct_change(), result.index_levels.pct_change()

    @pytest.mark.parametrize("scheme", [EqualWeighted(), MarketCapWeighted()])
    def test_identity_holds_to_machine_precision(self,
                                                 scheme):
        weights, asset_returns, index_returns = self._weights_and_returns(scheme)

        reconstructed = (weights.shift(1) * asset_returns).sum(axis=1)
        difference = (index_returns - reconstructed).dropna().abs()

        assert difference.max() < 1e-12, f"max error {difference.max():.3e}"

    def test_weights_sum_to_one_every_day(self):
        weights, _, _ = self._weights_and_returns(EqualWeighted())

        assert np.allclose(weights.sum(axis=1), 1.0)

    def test_equal_weighting_starts_equal_then_drifts(self):
        weights, _, _ = self._weights_and_returns(EqualWeighted())

        assert weights.iloc[0]["AAA"] == pytest.approx(0.5)
        assert weights.iloc[-1]["AAA"] > 0.5   # the winner grows its share


class TestDivisorContinuityStillHolds:

    def test_level_is_continuous_across_a_rebalance(self):
        fetcher = build_fetcher()
        calculator = IndexCalculator(definition(EqualWeighted(), "MONTHLY"), fetcher)
        result = calculator.run(start_date=START, end_date=END)

        returns = result.index_levels.pct_change().dropna()
        rebalances = [d for d in result.weight_snapshots if d in returns.index]

        assert rebalances, "expected rebalances inside the window"
        for date in rebalances:
            # A reconstitution must not create a jump; with these smooth price
            # paths every daily move is small.
            assert abs(returns.loc[date]) < 0.05

    def test_divisor_is_positive_throughout(self):
        result = IndexCalculator(definition(EqualWeighted(), "MONTHLY"),
                                 build_fetcher()).run(start_date=START, end_date=END)

        assert (result.divisor_history > 0).all()
