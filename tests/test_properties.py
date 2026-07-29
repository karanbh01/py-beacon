# tests/test_properties.py
"""Property-based tests for invariants that hold across Beacon's core layers.

Uses Hypothesis to generate synthetic, deterministic inputs (no network, no
fixture files) and checks the mathematical properties each component is
documented to guarantee:

* weighting schemes (``EqualWeighted`` / ``MarketCapWeighted``) produce
  weights that sum to one and are proportional to market capitalisation;
* ``IndexCalculator.adjust_divisor_for_rebalance`` preserves index-level
  continuity across a rebalance;
* ``Portfolio`` cash accounting reconciles across affordable trade sequences;
* ``IndexResult``/``BacktestResult`` return series recompound back to the
  original level series;
* the cost-of-carry pricing functions invert cleanly and are monotonic in
  their rate inputs.

Tolerances are chosen to reflect genuine floating-point accumulation (a
handful of additions/divisions/multiplications, so relative error stays many
orders of magnitude below 1e-9) rather than to paper over any real
discrepancy. Where a documented invariant did not hold as literally stated,
the test below asserts the property that actually does hold and the
discrepancy is called out in the accompanying report, not silently patched
away by loosening a tolerance.
"""
import math

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from beacon.asset.equity import Equity
from beacon.backtest.result import BacktestResult
from beacon.derivatives.pricing import cost_of_carry_fair_value, implied_repo_rate
from beacon.index.calculation import IndexCalculator
from beacon.index.methodology import EqualWeighted, MarketCapWeighted
from beacon.index.result import IndexResult
from beacon.portfolio.base import Portfolio

# ---------------------------------------------------------------------------
# Shared strategies and helpers
# ---------------------------------------------------------------------------
_PRICE = st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False)
_SHARES = st.floats(min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False)
_QUANTITY = st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
_RATE = st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False)
_TENOR = st.floats(min_value=1e-3, max_value=30.0, allow_nan=False, allow_infinity=False)
_STEP_MULTIPLIER = st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False)

_AS_OF = pd.Timestamp("2025-01-02")


@st.composite
def _level_paths(draw,
                 min_size: int,
                 max_size: int) -> list[float]:
    """Generate a positive level series via bounded day-over-day multipliers.

    Sampling each level independently from a wide absolute range (as
    ``_PRICE`` does) lets adjacent levels differ by many orders of
    magnitude. Reconstructing such a jump from a percentage return suffers
    genuine floating-point cancellation: the return is close to -100%, so
    ``1 + return`` cancels to a value with only absolute (not relative)
    precision, and dividing that back into a tiny level amplifies the
    error further. Real index/NAV level paths move gradually rather than
    collapsing several orders of magnitude in one step, so bounding the
    multiplier keeps this test in the regime the round-trip identity is
    actually well-conditioned for (see the accompanying report for the
    concrete failing case this was found from).
    """
    base = draw(_PRICE)
    length = draw(st.integers(min_value=min_size, max_value=max_size))
    multipliers = draw(st.lists(_STEP_MULTIPLIER, min_size=length - 1, max_size=length - 1))

    values = [base]
    for multiplier in multipliers:
        values.append(values[-1] * multiplier)

    return values


class _FakeMarketDataProvider:
    """Minimal in-memory stand-in for ``DataFetcher``.

    Exposes only the methods ``MarketCapWeighted.calculate_weights`` calls,
    mirroring the shapes documented on ``beacon.data.fetcher.DataFetcher``:
    ``fetch_market_data`` returns a single-identifier, date-indexed frame
    with a ``CLOSE`` column, and ``fetch_shares_outstanding`` returns a raw
    float. ``EqualWeighted`` never calls the provider at all.
    """
    def __init__(self,
                 prices: dict[str, float],
                 shares: dict[str, float]):
        self._prices = prices
        self._shares = shares

    def fetch_market_data(self,
                          identifier: str,
                          start_date: str | None = None,
                          end_date: str | None = None,
                          columns: list[str] | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            {"CLOSE": [self._prices[identifier]]},
            index=pd.DatetimeIndex([_AS_OF], name="DATE"),
        )

    def fetch_shares_outstanding(self,
                                 identifier: str,
                                 date: str) -> float | None:
        return self._shares[identifier]

    def fetch_free_float_factor(self,
                                identifier: str,
                                date: str) -> float | None:
        return None


def _make_equities(count: int) -> list[Equity]:
    """Build *count* distinct synthetic Equity constituents."""
    return [
        Equity(name=f"Asset{i}", currency="USD", ticker=f"TICK{i}", exchange="NYSE")
        for i in range(count)
    ]


def _prices_and_shares(equities: list[Equity],
                       data: list[tuple[float, float]]) -> tuple[dict, dict]:
    """Zip generated (price, shares) pairs onto *equities* by ticker."""
    prices = {e.ticker: price for e, (price, _shares) in zip(equities, data, strict=True)}
    shares = {e.ticker: shares for e, (_price, shares) in zip(equities, data, strict=True)}
    return prices, shares


# ---------------------------------------------------------------------------
# Weighting schemes: EqualWeighted / MarketCapWeighted
# ---------------------------------------------------------------------------

class TestWeightingSchemes:

    @given(st.lists(st.tuples(_PRICE, _SHARES), min_size=1, max_size=8))
    @settings(max_examples=50, deadline=None)
    def test_equal_weighted_uniform_and_sums_to_one(self,
                                                    data):
        equities = _make_equities(len(data))
        prices, shares = _prices_and_shares(equities, data)
        provider = _FakeMarketDataProvider(prices, shares)

        weights = EqualWeighted().calculate_weights(equities, _AS_OF, provider)

        assert set(weights) == set(equities)
        assert math.isclose(sum(weights.values()), 1.0, rel_tol=1e-9)

        expected = 1.0 / len(data)
        for weight in weights.values():
            assert weight >= 0.0
            assert math.isclose(weight, expected, rel_tol=1e-9)

    @given(st.lists(st.tuples(_PRICE, _SHARES), min_size=1, max_size=8))
    @settings(max_examples=50, deadline=None)
    def test_market_cap_weighted_nonnegative_and_sums_to_one(self,
                                                             data):
        equities = _make_equities(len(data))
        prices, shares = _prices_and_shares(equities, data)
        provider = _FakeMarketDataProvider(prices, shares)

        weights = MarketCapWeighted().calculate_weights(equities, _AS_OF, provider)

        assert set(weights) == set(equities)
        assert math.isclose(sum(weights.values()), 1.0, rel_tol=1e-9)
        assert all(weight >= 0.0 for weight in weights.values())

    @given(st.lists(st.tuples(_PRICE, _SHARES), min_size=2, max_size=8))
    @settings(max_examples=50, deadline=None)
    def test_market_cap_weighted_proportional_to_cap(self,
                                                     data):
        equities = _make_equities(len(data))
        prices, shares = _prices_and_shares(equities, data)
        provider = _FakeMarketDataProvider(prices, shares)

        weights = MarketCapWeighted().calculate_weights(equities, _AS_OF, provider)
        caps = {e: prices[e.ticker] * shares[e.ticker] for e in equities}

        for i, asset_i in enumerate(equities):
            for asset_j in equities[i + 1:]:
                weight_ratio = weights[asset_i] / weights[asset_j]
                cap_ratio = caps[asset_i] / caps[asset_j]
                assert math.isclose(weight_ratio, cap_ratio, rel_tol=1e-9)

    @given(st.lists(st.tuples(_PRICE, _SHARES), min_size=1, max_size=8),
           st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=None)
    def test_market_cap_weighted_scale_invariant(self,
                                                 data,
                                                 scale):
        equities = _make_equities(len(data))
        prices, shares = _prices_and_shares(equities, data)
        weights = MarketCapWeighted().calculate_weights(
            equities, _AS_OF, _FakeMarketDataProvider(prices, shares))

        scaled_prices = {ticker: price * scale for ticker, price in prices.items()}
        scaled_weights = MarketCapWeighted().calculate_weights(
            equities, _AS_OF, _FakeMarketDataProvider(scaled_prices, shares))

        for asset in equities:
            assert math.isclose(weights[asset], scaled_weights[asset], rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Divisor continuity: IndexCalculator.adjust_divisor_for_rebalance
# ---------------------------------------------------------------------------

class TestDivisorContinuity:

    @given(st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.01, max_value=1e9, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.01, max_value=1e9, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200, deadline=None)
    def test_level_continuity_across_rebalance(self,
                                               old_divisor,
                                               old_market_value,
                                               new_market_value):
        new_divisor = IndexCalculator.adjust_divisor_for_rebalance(
            old_divisor, old_market_value, new_market_value)

        level_before = old_market_value / old_divisor
        level_after = new_market_value / new_divisor

        # A handful of divisions/multiplications: relative error stays near
        # machine epsilon, so 1e-9 is many orders of magnitude looser than
        # what actually accumulates.
        assert math.isclose(level_before, level_after, rel_tol=1e-9)

    @given(st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
           st.floats(min_value=-1e6, max_value=0.0, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=None)
    def test_nonpositive_old_market_value_raises(self,
                                                 old_divisor,
                                                 old_market_value,
                                                 new_market_value):
        with pytest.raises(ValueError, match="old_market_value must be positive"):
            IndexCalculator.adjust_divisor_for_rebalance(
                old_divisor, old_market_value, new_market_value)


# ---------------------------------------------------------------------------
# Portfolio accounting: beacon.portfolio.base.Portfolio
# ---------------------------------------------------------------------------

class TestPortfolioAccounting:

    @given(st.lists(st.tuples(_PRICE, _QUANTITY), min_size=1, max_size=5))
    @settings(max_examples=50, deadline=None)
    def test_affordable_buys_reconcile_cash(self,
                                            trades):
        cost = 1.0
        total_needed = sum(price * quantity + cost for price, quantity in trades)
        initial_cash = total_needed + 1.0  # margin: every prefix stays affordable

        portfolio = Portfolio("p", initial_cash=initial_cash)
        for i, (price, quantity) in enumerate(trades):
            portfolio.execute_buy(f"A{i}", quantity=quantity, price=price, cost=cost)

        expected_cash = initial_cash - total_needed
        # cash_balance is produced by a chain of sequential subtractions
        # inside execute_buy, while expected_cash is one up-front sum then a
        # single subtraction — a different order of operations that can
        # leave a residual expected_cash close to zero even when the trade
        # notionals are large (e.g. ~1e8). rel_tol alone then compares
        # against a near-zero reference, so the tolerance is anchored to the
        # notional scale actually flowing through the portfolio instead.
        assert math.isclose(portfolio.cash_balance, expected_cash,
                            rel_tol=1e-9, abs_tol=1e-9 * initial_cash)

    @given(_PRICE, _QUANTITY)
    @settings(max_examples=50, deadline=None)
    def test_buy_then_full_sell_returns_to_initial_cash(self,
                                                        price,
                                                        quantity):
        initial_cash = price * quantity + 1.0  # margin covers the buy
        portfolio = Portfolio("p", initial_cash=initial_cash)

        portfolio.execute_buy("A", quantity=quantity, price=price, cost=0.0)
        portfolio.execute_sell("A", quantity=quantity, price=price, cost=0.0)

        assert math.isclose(portfolio.cash_balance, initial_cash, rel_tol=1e-9)
        assert "A" not in portfolio.holdings

    @given(st.lists(st.tuples(_PRICE, _QUANTITY), min_size=1, max_size=5),
           st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=None)
    def test_weights_bounded_in_unit_interval(self,
                                              holdings,
                                              extra_cash):
        total_needed = sum(price * quantity for price, quantity in holdings)
        initial_cash = total_needed + extra_cash + 1.0  # margin

        portfolio = Portfolio("p", initial_cash=initial_cash)
        for i, (price, quantity) in enumerate(holdings):
            portfolio.execute_buy(f"A{i}", quantity=quantity, price=price, cost=0.0)

        for weight in portfolio.get_weights().values():
            assert 0.0 <= weight <= 1.0

    @given(st.lists(st.tuples(_PRICE, _QUANTITY), min_size=1, max_size=5))
    @settings(max_examples=50, deadline=None)
    def test_weights_sum_to_one_when_fully_invested(self,
                                                    holdings):
        """get_weights() sums to 1.0 only in the fully-invested special case.

        ``get_weights()`` returns holdings' shares of total value (holdings +
        cash) but has no entry for cash itself, so any leftover cash balance
        keeps the returned weights from summing to 1.0 in general — only
        ``get_holdings_summary()`` (which adds an explicit CASH row) does
        that unconditionally. This test drains cash to exactly zero (the one
        case where the two coincide) by sweeping the remainder into one more
        holding, rather than loosening the assertion.
        """
        total_needed = sum(price * quantity for price, quantity in holdings)
        initial_cash = total_needed + 1000.0  # ample margin, swept below

        portfolio = Portfolio("p", initial_cash=initial_cash)
        for i, (price, quantity) in enumerate(holdings):
            portfolio.execute_buy(f"A{i}", quantity=quantity, price=price, cost=0.0)

        remaining = portfolio.cash_balance
        if remaining > 0.0:
            portfolio.execute_buy("REMAINDER", quantity=1.0, price=remaining, cost=0.0)

        weights = portfolio.get_weights()
        assert math.isclose(portfolio.cash_balance, 0.0, abs_tol=1e-6)
        assert math.isclose(sum(weights.values()), 1.0, rel_tol=1e-9)
        for weight in weights.values():
            assert 0.0 <= weight <= 1.0


# ---------------------------------------------------------------------------
# Return/level round-trip: IndexResult.get_returns / BacktestResult.get_returns
# ---------------------------------------------------------------------------

class TestReturnLevelRoundTrip:

    @given(_level_paths(min_size=2, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_index_result_returns_recompound_to_levels(self,
                                                       values):
        dates = pd.bdate_range("2025-01-02", periods=len(values))
        levels = pd.Series(values, index=dates)

        result = IndexResult(
            index_id="TEST",
            index_levels=levels,
            divisor_history=levels.copy(),
            constituent_snapshots={},
            weight_snapshots={},
        )

        returns = result.get_returns()
        recompounded = levels.iloc[0] * (1.0 + returns).cumprod()

        assert len(recompounded) == len(levels) - 1
        # A short multiplicative chain (<=19 terms): relative error stays
        # orders of magnitude below 1e-9.
        for original, rebuilt in zip(levels.iloc[1:], recompounded, strict=True):
            assert math.isclose(original, rebuilt, rel_tol=1e-9)

    @given(_level_paths(min_size=2, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_backtest_result_returns_recompound_to_levels(self,
                                                          values):
        dates = pd.bdate_range("2025-01-02", periods=len(values))
        nav = pd.Series(values, index=dates)

        result = BacktestResult(
            portfolio_id="TEST",
            initial_capital=nav.iloc[0],
            portfolio_nav=nav,
            cash_history=nav.copy(),
            transactions=[],
            actual_weight_history=pd.DataFrame(index=dates),
        )

        returns = result.get_returns()
        recompounded = nav.iloc[0] * (1.0 + returns).cumprod()

        for original, rebuilt in zip(nav.iloc[1:], recompounded, strict=True):
            assert math.isclose(original, rebuilt, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Cost-of-carry pricing: beacon.derivatives.pricing
# ---------------------------------------------------------------------------

class TestCostOfCarryPricing:

    @given(st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
           _RATE, _RATE, _TENOR)
    @settings(max_examples=200, deadline=None)
    def test_implied_repo_rate_inverts_fair_value(self,
                                                  spot,
                                                  risk_free_rate,
                                                  dividend_yield,
                                                  tenor):
        fair_value = cost_of_carry_fair_value(spot, risk_free_rate, dividend_yield, tenor)
        recovered_rate = implied_repo_rate(fair_value, spot, dividend_yield, tenor)

        # exp() then log() round-trip plus one division: error stays near
        # machine epsilon regardless of tenor, so 1e-9 is a loose bound.
        assert math.isclose(recovered_rate, risk_free_rate, rel_tol=1e-9, abs_tol=1e-9)

    @given(st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
           _RATE, _RATE, _TENOR,
           st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200, deadline=None)
    def test_fair_value_nondecreasing_in_risk_free_rate(self,
                                                        spot,
                                                        risk_free_rate,
                                                        dividend_yield,
                                                        tenor,
                                                        bump):
        lower = cost_of_carry_fair_value(spot, risk_free_rate, dividend_yield, tenor)
        higher = cost_of_carry_fair_value(spot, risk_free_rate + bump, dividend_yield, tenor)

        assert higher >= lower - 1e-9  # tolerance for float noise near bump == 0

    @given(st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
           _RATE, _RATE, _TENOR,
           st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200, deadline=None)
    def test_fair_value_nonincreasing_in_dividend_yield(self,
                                                        spot,
                                                        risk_free_rate,
                                                        dividend_yield,
                                                        tenor,
                                                        bump):
        base = cost_of_carry_fair_value(spot, risk_free_rate, dividend_yield, tenor)
        bumped = cost_of_carry_fair_value(spot, risk_free_rate, dividend_yield + bump, tenor)

        assert base >= bumped - 1e-9  # tolerance for float noise near bump == 0

    @given(st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
           _RATE, _TENOR)
    @settings(max_examples=100, deadline=None)
    def test_fair_value_equals_spot_when_rate_equals_yield(self,
                                                           spot,
                                                           rate,
                                                           tenor):
        fair_value = cost_of_carry_fair_value(spot, rate, rate, tenor, borrow_cost=0.0)
        assert math.isclose(fair_value, spot, rel_tol=1e-9)
