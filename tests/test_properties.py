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

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from beacon.analysis import attribute
from beacon.asset.equity import Equity
from beacon.backtest.result import BacktestResult
from beacon.derivatives.pricing import cost_of_carry_fair_value, implied_repo_rate
from beacon.index.calculation import IndexCalculator
from beacon.index.capping import TOLERANCE, apply_cap
from beacon.index.methodology import EqualWeighted, MarketCapWeighted
from beacon.index.result import IndexResult
from beacon.optimise import (
    FullInvestment,
    PositionBounds,
    efficient_frontier,
    minimise_tracking_error,
)
from beacon.portfolio.base import Portfolio
from beacon.risk import estimate_risk_model
from beacon.risk.factors import fit_factor_model, z_scores

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

@st.composite
def _weights_and_cap(draw,
                     min_size: int = 2,
                     max_size: int = 12) -> tuple[dict[str, float], float]:
    """Generate normalised weights alongside a cap that is feasible for them.

    The cap is drawn from ``[1/n, 1]`` because anything below ``1/n`` is
    genuinely impossible to satisfy — every name would sit at the cap and the
    total would still fall short of 1.0. Infeasible caps are covered by their
    own test in tests/test_capping.py, which asserts they raise.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    raw = draw(st.lists(st.floats(min_value=0.01, max_value=1e4,
                                  allow_nan=False, allow_infinity=False),
                        min_size=size, max_size=size))
    total = sum(raw)
    weights = {f"A{i}": value / total for i, value in enumerate(raw)}

    cap = draw(st.floats(min_value=1.0 / size, max_value=1.0,
                         allow_nan=False, allow_infinity=False))

    return weights, cap


class TestCapping:
    """The two invariants BN-57 had to defer until capping existed."""

    @given(_weights_and_cap())
    @settings(max_examples=200, deadline=None)
    def test_weights_sum_to_one_after_capping(self,
                                              case):
        weights, cap = case

        capped, _ = apply_cap(weights, cap)

        assert math.isclose(sum(capped.values()), 1.0, rel_tol=1e-9)

    @given(_weights_and_cap())
    @settings(max_examples=200, deadline=None)
    def test_no_weight_exceeds_the_cap(self,
                                       case):
        weights, cap = case

        capped, _ = apply_cap(weights, cap)

        assert max(capped.values()) <= cap * (1 + TOLERANCE)

    @given(_weights_and_cap())
    @settings(max_examples=200, deadline=None)
    def test_iteration_converges(self,
                                 case):
        """Each pass caps at least one more name, so it cannot run away."""
        weights, cap = case

        _, report = apply_cap(weights, cap)

        assert report.passes <= len(weights)

    @given(_weights_and_cap())
    @settings(max_examples=200, deadline=None)
    def test_capping_never_creates_a_negative_weight(self,
                                                     case):
        weights, cap = case

        capped, _ = apply_cap(weights, cap)

        assert all(weight >= 0.0 for weight in capped.values())

    @given(_weights_and_cap())
    @settings(max_examples=200, deadline=None)
    def test_redistribution_matches_what_was_taken(self,
                                                   case):
        """Weight is conserved: what comes off capped names goes to the rest."""
        weights, cap = case

        capped, report = apply_cap(weights, cap)

        gained = sum(capped[name] - weights[name]
                     for name in weights if name not in report.capped)

        assert math.isclose(gained, report.redistributed, rel_tol=1e-6, abs_tol=1e-12)

    @given(_weights_and_cap())
    @settings(max_examples=100, deadline=None)
    def test_capping_is_idempotent(self,
                                   case):
        """Capping already-capped weights changes nothing."""
        weights, cap = case

        once, _ = apply_cap(weights, cap)
        twice, report = apply_cap(once, cap)

        assert report.was_capped is False
        for name, weight in once.items():
            assert math.isclose(twice[name], weight, rel_tol=1e-9)


@st.composite
def _returns_panels(draw,
                    min_assets: int = 1,
                    max_assets: int = 6,
                    min_observations: int = 2,
                    max_observations: int = 40) -> pd.DataFrame:
    """Generate a returns panel, deliberately including degenerate shapes.

    Observations are drawn independently of the asset count, so panels with
    fewer periods than assets — where the sample covariance is singular and
    shrinkage matters most — turn up regularly rather than being excluded.
    """
    assets = draw(st.integers(min_value=min_assets, max_value=max_assets))
    observations = draw(st.integers(min_value=min_observations,
                                    max_value=max_observations))

    values = draw(st.lists(
        st.lists(st.floats(min_value=-0.5, max_value=0.5,
                           allow_nan=False, allow_infinity=False),
                 min_size=assets, max_size=assets),
        min_size=observations, max_size=observations))

    return pd.DataFrame(values, columns=[f"A{i}" for i in range(assets)])


class TestRiskModelInvariants:
    """The PSD invariant BN-57 had to defer until a risk model existed."""

    @given(_returns_panels())
    @settings(max_examples=200, deadline=None)
    def test_shrunk_covariance_is_positive_semi_definite(self,
                                                         panel):
        """Holds for any panel, including fewer observations than assets.

        Both the sample covariance and the shrinkage target are PSD, and the
        estimate is a convex combination of them, so PSD is preserved by
        construction rather than by luck.
        """
        model = estimate_risk_model(panel)

        assert model.diagnostics.positive_semi_definite

    @given(_returns_panels())
    @settings(max_examples=200, deadline=None)
    def test_correlation_diagonal_is_exactly_one(self,
                                                panel):
        correlation = estimate_risk_model(panel).correlation.to_numpy()

        assert (np.diag(correlation) == 1.0).all()

    @given(_returns_panels())
    @settings(max_examples=200, deadline=None)
    def test_matrices_are_symmetric(self,
                                    panel):
        model = estimate_risk_model(panel)

        for matrix in (model.covariance.to_numpy(), model.correlation.to_numpy()):
            assert (matrix == matrix.T).all()

    @given(_returns_panels())
    @settings(max_examples=200, deadline=None)
    def test_correlations_stay_within_minus_one_and_one(self,
                                                        panel):
        correlation = estimate_risk_model(panel).correlation.to_numpy()

        assert correlation.min() >= -1.0 - 1e-9
        assert correlation.max() <= 1.0 + 1e-9

    @given(_returns_panels(min_assets=2))
    @settings(max_examples=200, deadline=None)
    def test_portfolio_variance_is_never_negative(self,
                                                 panel):
        """A PSD covariance cannot produce a negative variance."""
        model = estimate_risk_model(panel)
        weights = dict.fromkeys(model.asset_ids, 1.0 / len(model.asset_ids))

        assert model.portfolio_variance(weights) >= 0.0

    @given(_returns_panels(min_assets=2))
    @settings(max_examples=100, deadline=None)
    def test_variances_are_non_negative(self,
                                        panel):
        diagonal = np.diag(estimate_risk_model(panel).covariance.to_numpy())

        assert (diagonal >= -1e-12).all()

    @given(_returns_panels())
    @settings(max_examples=100, deadline=None)
    def test_zero_intensity_leaves_the_sample_covariance_psd(self,
                                                             panel):
        """The unshrunk estimate is PSD too — shrinkage is not what rescues it."""
        model = estimate_risk_model(panel, intensity=0.0)

        assert model.diagnostics.positive_semi_definite


@st.composite
def _attribution_inputs(draw,
                        min_assets: int = 2,
                        max_assets: int = 5,
                        min_periods: int = 2,
                        max_periods: int = 40):
    """Generate weights and asset returns for an attribution.

    Weights are normalised per period so they sum to 1, and returns are bounded
    away from -100% so the linking coefficient stays defined — an index that
    goes to zero has no attribution, which the code raises on rather than
    fudging.
    """
    assets = draw(st.integers(min_value=min_assets, max_value=max_assets))
    periods = draw(st.integers(min_value=min_periods, max_value=max_periods))
    names = [f"A{i}" for i in range(assets)]
    dates = pd.bdate_range("2024-01-01", periods=periods)

    raw = draw(st.lists(
        st.lists(st.floats(min_value=0.01, max_value=10.0,
                           allow_nan=False, allow_infinity=False),
                 min_size=assets, max_size=assets),
        min_size=periods, max_size=periods))
    weights = pd.DataFrame(raw, index=dates, columns=names)
    weights = weights.div(weights.sum(axis=1), axis=0)

    returns = draw(st.lists(
        st.lists(st.floats(min_value=-0.3, max_value=0.3,
                           allow_nan=False, allow_infinity=False),
                 min_size=assets, max_size=assets),
        min_size=periods, max_size=periods))
    asset_returns = pd.DataFrame(returns, index=dates, columns=names)

    period_returns = (weights.shift(1) * asset_returns).sum(axis=1)

    return period_returns, weights, asset_returns


class TestAttributionInvariants:
    """BN-57's last deferred invariant: attribution reconciles."""

    @given(_attribution_inputs())
    @settings(max_examples=200, deadline=None)
    def test_contributions_sum_to_the_total_return(self,
                                                   case):
        """Exact after Carino linking, for any weights and returns."""
        result = attribute(*case)

        assert result.explained == pytest.approx(result.total_return, abs=1e-9)

    @given(_attribution_inputs())
    @settings(max_examples=200, deadline=None)
    def test_the_residual_is_negligible(self,
                                        case):
        result = attribute(*case)

        assert result.reconciles(), f"residual {result.residual:.3e}"

    @given(_attribution_inputs())
    @settings(max_examples=200, deadline=None)
    def test_every_constituent_is_accounted_for(self,
                                                case):
        """No constituent may be dropped — its contribution would vanish."""
        _, weights, _ = case
        result = attribute(*case)

        assert {item.asset_id for item in result.contributions} == set(weights.columns)

    @given(_attribution_inputs())
    @settings(max_examples=100, deadline=None)
    def test_linking_preserves_the_sign_of_the_total(self,
                                                    case):
        """A positive total cannot decompose into a negative explained sum."""
        result = attribute(*case)

        if abs(result.total_return) > 1e-9:
            assert math.copysign(1.0, result.explained) == math.copysign(
                1.0, result.total_return)


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


# --- BN-92: constrained tracking-error minimisation -------------------------

_TARGET_WEIGHTS = st.lists(
    st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=8)


def _normalised(raw: list[float]) -> pd.Series:
    """A target weight vector summing to one."""
    total = sum(raw)

    return pd.Series({f"A{index}": value / total
                      for index, value in enumerate(raw)})


class TestOptimiserProperties:
    """Invariants the tracking optimiser guarantees for any feasible problem."""

    @given(_TARGET_WEIGHTS)
    @settings(max_examples=100, deadline=None)
    def test_a_feasible_target_is_reproduced(self,
                                             raw):
        """With nothing binding, the closest feasible portfolio is the target."""
        target = _normalised(raw)

        result = minimise_tracking_error(target, [FullInvestment()])

        assert (result.weights - target).abs().max() < 1e-7

    @given(_TARGET_WEIGHTS)
    @settings(max_examples=100, deadline=None)
    def test_the_solution_is_always_fully_invested(self,
                                                   raw):
        target = _normalised(raw)
        cap = max(float(target.max()) / 2.0, 1.0 / len(target))

        result = minimise_tracking_error(
            target, [FullInvestment(), PositionBounds(0.0, cap)])

        assert abs(float(result.weights.sum()) - 1.0) < 1e-7

    @given(_TARGET_WEIGHTS,
           st.floats(min_value=0.0, max_value=0.4, allow_nan=False,
                     allow_infinity=False))
    @settings(max_examples=150, deadline=None)
    def test_tightening_a_cap_cannot_improve_the_objective(self,
                                                           raw,
                                                           squeeze):
        """Shrinking the feasible set can only move the optimum further away.

        A genuine theorem rather than a smoke test: every portfolio allowed
        under the tighter cap was already allowed under the looser one, so the
        looser problem's minimum is taken over a superset and cannot be worse.
        An optimiser that quietly relaxed a constraint, or that returned a
        local rather than global optimum, would break this.
        """
        target = _normalised(raw)
        floor = 1.0 / len(target)

        loose = max(float(target.max()), floor)
        tight = max(loose * (1.0 - squeeze), floor)

        looser = minimise_tracking_error(
            target, [FullInvestment(), PositionBounds(0.0, loose)])
        tighter = minimise_tracking_error(
            target, [FullInvestment(), PositionBounds(0.0, tight)])

        assert tighter.diagnostics.objective >= looser.diagnostics.objective - 1e-9

    @given(_TARGET_WEIGHTS)
    @settings(max_examples=100, deadline=None)
    def test_no_returned_solution_ever_violates_its_constraints(self,
                                                                raw):
        """The contract: an answer comes back only if it satisfies everything."""
        target = _normalised(raw)
        cap = max(float(target.max()) * 0.8, 1.0 / len(target))

        result = minimise_tracking_error(
            target, [FullInvestment(), PositionBounds(0.0, cap)])

        assert not any(slack.is_violated for slack in result.slacks)
        assert float(result.weights.max()) <= cap + 1e-7
        assert float(result.weights.min()) >= -1e-7


# --- BN-93: frontier and factor-model invariants ----------------------------

_POSITIVE_VOLATILITIES = st.lists(
    st.floats(min_value=0.005, max_value=0.05, allow_nan=False,
              allow_infinity=False),
    min_size=3,
    max_size=6)

_ACTIVE_BETS = st.lists(
    st.floats(min_value=-0.15, max_value=0.15, allow_nan=False,
              allow_infinity=False),
    min_size=4,
    max_size=8)


def _risk_model_from(volatilities: list[float], seed: int = 5):
    """A risk model over independent assets with the given volatilities."""
    generator = np.random.default_rng(seed)
    periods = 300

    panel = pd.DataFrame(
        {f"A{index}": generator.normal(0.0, volatility, periods)
         for index, volatility in enumerate(volatilities)},
        index=pd.bdate_range("2024-01-01", periods=periods))

    return estimate_risk_model(panel, intensity=0.1)


class TestFrontierProperties:
    """Invariants the efficient frontier guarantees for any feasible problem."""

    @given(_POSITIVE_VOLATILITIES)
    @settings(max_examples=25, deadline=None)
    def test_risk_never_falls_as_return_rises(self,
                                              volatilities):
        """The defining property of a frontier, on arbitrary universes.

        Insisting on more return can only shrink the feasible set, so the least
        risk available cannot improve. A point that solved to a local rather
        than global optimum would show up as a dip in the curve.
        """
        model = _risk_model_from(volatilities)
        returns = {asset: 0.02 + 0.02 * index
                   for index, asset in enumerate(model.covariance.index)}

        frontier = efficient_frontier(model, returns, points=5)

        assert frontier.is_monotonic()

    @given(_POSITIVE_VOLATILITIES)
    @settings(max_examples=25, deadline=None)
    def test_no_frontier_point_beats_the_minimum_variance_portfolio(self,
                                                                    volatilities):
        model = _risk_model_from(volatilities)
        returns = {asset: 0.02 + 0.02 * index
                   for index, asset in enumerate(model.covariance.index)}

        frontier = efficient_frontier(model, returns, points=5)

        assert min(frontier.volatilities) >= frontier.minimum_variance.volatility - 1e-9

    @given(_POSITIVE_VOLATILITIES)
    @settings(max_examples=25, deadline=None)
    def test_the_tangency_is_the_best_sharpe_ratio_found(self,
                                                         volatilities):
        model = _risk_model_from(volatilities)
        returns = {asset: 0.02 + 0.02 * index
                   for index, asset in enumerate(model.covariance.index)}

        frontier = efficient_frontier(model, returns, points=5, risk_free_rate=0.01)
        best = max(point.sharpe_ratio for point in frontier.points)

        assert frontier.tangency.sharpe_ratio >= best - 1e-9


class TestActiveRiskIdentity:
    """TE² = factor + specific, for any active position at all."""

    @given(_ACTIVE_BETS)
    @settings(max_examples=100, deadline=None)
    def test_the_decomposition_always_reconciles(self,
                                                 bets):
        """Exact by construction, so this holds for arbitrary bets.

        Not a tolerance question: Sigma is *defined* as BFBᵀ + D, so the split
        is algebra rather than approximation. Anything above float noise means
        the algebra is wrong.
        """
        assets = [f"A{index}" for index in range(len(bets))]
        generator = np.random.default_rng(31)

        exposures = z_scores(pd.DataFrame(
            {"value": generator.normal(0.0, 1.0, len(assets)),
             "momentum": generator.normal(0.0, 1.0, len(assets))},
            index=assets))

        panel = pd.DataFrame(
            generator.normal(0.0, 0.01, (200, len(assets))),
            index=pd.bdate_range("2024-01-01", periods=200),
            columns=assets)

        model = fit_factor_model(panel, exposures)

        benchmark = dict.fromkeys(assets, 1.0 / len(assets))
        # Centred so the active position sums to zero, as a real one does.
        centred = np.array(bets) - np.mean(bets)
        weights = {asset: benchmark[asset] + bet
                   for asset, bet in zip(assets, centred, strict=True)}

        decomposition = model.decompose_active_risk(weights, benchmark)

        assert abs(decomposition.residual) < 1e-15
        assert decomposition.factor_variance >= 0.0
        assert decomposition.specific_variance >= 0.0
