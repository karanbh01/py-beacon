# tests/test_derivatives_valuation.py
"""BN-417: futures, swaps and rolls valued from their terms in the library,
as the server's derivatives endpoints used to do themselves."""
import math

import pandas as pd
import pytest

from beacon.derivatives.curves import RateCurve
from beacon.derivatives.pricing import cost_of_carry_fair_value
from beacon.derivatives.swaps import TotalReturnSwap
from beacon.derivatives.valuation import (
    carry_of,
    financing_schedule,
    roll_between,
    tenor_of,
    value_future,
    value_swap,
    years_between,
)
from beacon.exceptions import InvalidArgumentError

SPOT = 5000.0
FLAT = RateCurve.flat(0.05)


def swap() -> TotalReturnSwap:
    return TotalReturnSwap(derivative_id="T1", underlying_id="SPX", currency="USD",
                           start_date="2024-01-02", end_date="2024-12-31", notional=1e6,
                           spread_bps=50.0, reference_rate="SOFR",
                           payment_frequency="QUARTERLY")


class TestTenor:

    def test_dates_win_over_a_given_tenor(self):
        assert tenor_of("2024-01-01", "2024-12-31", 3.0) == pytest.approx(365 / 365)
        assert tenor_of(None, None, 0.5) == 0.5

    def test_refused(self):
        with pytest.raises(InvalidArgumentError, match="before the valuation date"):
            tenor_of("2024-07-01", "2024-01-01", None)
        with pytest.raises(InvalidArgumentError, match="time to expiry is needed"):
            tenor_of(None, None, None)
        with pytest.raises(InvalidArgumentError, match="must not be negative"):
            tenor_of(None, None, -1.0)

    def test_years_between(self):
        assert years_between(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-07-01")) == 182 / 365


class TestFuture:

    def test_fair_value_is_the_cost_of_carry(self):
        found = value_future(SPOT, 0.5, FLAT, dividend_yield=0.015, borrow_cost=0.002)

        assert found.fair_value == pytest.approx(SPOT * math.exp((0.05 - 0.015 + 0.002) * 0.5))
        assert found.financing_rate == pytest.approx(0.05)
        assert found.basis is None and found.implied_repo is None

    def test_the_carry_parts_and_their_residual(self):
        carry = carry_of(SPOT, 0.05, 0.5, 0.015, 0.0,
                         cost_of_carry_fair_value(SPOT, 0.05, 0.015, 0.5))

        assert carry.financing == pytest.approx(SPOT * math.expm1(0.025))
        assert carry.dividend == pytest.approx(-SPOT * math.expm1(0.0075))
        assert carry.total == pytest.approx(carry.financing + carry.dividend + carry.borrow
                                            + carry.residual)

    def test_a_quote_gives_the_basis_and_implied_repo(self):
        found = value_future(SPOT, 0.5, FLAT, market_price=5150.0, contract_multiplier=50.0,
                             contracts=2.0)

        assert found.basis == pytest.approx(5150.0 - found.fair_value)
        assert found.implied_repo is not None
        assert found.contract_value == pytest.approx(found.fair_value * 100.0)

    def test_the_grid_defaults_around_the_contract(self):
        grid = value_future(SPOT, 0.5, FLAT).sensitivity

        assert grid.shape == (4, 5)

    def test_discrete_dividends_price_off_their_amounts(self):
        plain = value_future(SPOT, 0.5, FLAT)
        paid = value_future(SPOT, 0.5, FLAT, dividends=[(0.25, 20.0)])

        assert paid.fair_value < plain.fair_value


class TestSwap:

    def test_its_legs_and_schedule(self):
        valuation, reset = pd.Timestamp("2024-02-15"), pd.Timestamp("2024-01-02")
        found = value_swap(swap(), valuation, reset, FLAT, 0.05, 5100.0, 5000.0)

        assert found.accrual_days == 44
        assert found.present_value == pytest.approx(found.total_return_leg - found.financing_leg)
        assert found.fair_spread_bps is not None
        assert [period.days for period in found.schedule][:1] == [91]
        assert found.breakeven == []

    def test_nothing_accrued_has_no_fair_spread(self):
        day = pd.Timestamp("2024-01-02")

        assert value_swap(swap(), day, day, FLAT, 0.05, 5000.0, 5000.0).fair_spread_bps is None

    def test_later_periods_project_off_the_curve(self):
        curve = RateCurve.from_pillars({0.25: 0.04, 1.0: 0.06})
        rows = financing_schedule(swap(), pd.Timestamp("2024-02-15"),
                                  pd.Timestamp("2024-01-02"), curve, 0.05)

        assert rows[0].rate == 0.05
        assert rows[-1].rate != 0.05

    def test_breakeven_against_futures(self):
        found = value_swap(swap(), pd.Timestamp("2024-02-15"), pd.Timestamp("2024-01-02"),
                           FLAT, 0.05, 5100.0, 5000.0, futures_prices=[5100.0, 5200.0],
                           time_to_expiry=0.5)

        assert [row["futures_price"] for row in found.breakeven] == [5100.0, 5200.0]


class TestRoll:

    def test_the_carry_roll(self):
        found = roll_between(SPOT, pd.Timestamp("2024-01-02"), pd.Timestamp("2024-03-15"),
                             pd.Timestamp("2024-06-21"), FLAT)

        assert found.roll_cost == pytest.approx(found.back_price - found.front_price)
        assert found.roll_cost > 0.0

    def test_the_back_must_follow_the_front(self):
        with pytest.raises(InvalidArgumentError, match="must be after"):
            roll_between(SPOT, pd.Timestamp("2024-01-02"), pd.Timestamp("2024-06-21"),
                         pd.Timestamp("2024-03-15"), FLAT)
