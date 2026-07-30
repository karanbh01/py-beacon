# tests/test_derivatives_curves.py
"""BN-96: rate curves, TRS DV01, term structure and the sensitivity grid."""
import math

import pandas as pd
import pytest

from beacon.derivatives import (
    BASIS_POINT,
    FuturesQuote,
    RateCurve,
    TermStructure,
    TotalReturnSwap,
    sensitivity_grid,
)
from beacon.derivatives.pricing import cost_of_carry_fair_value
from beacon.exceptions import CalculationError

SPOT = 100.0
DIVIDEND_YIELD = 0.02
VALUATION_DATE = pd.Timestamp("2024-01-01")

# A curve with real shape, so interpolation has something to do.
PILLARS = {0.25: 0.030, 1.0: 0.040, 5.0: 0.045}

NOTIONAL = 10_000_000.0

# The two axes of the sensitivity grid.
GRID_TENORS = [0.25, 0.5, 1.0]
GRID_RATES = [0.03, 0.05, 0.07]


def swap(reset_type: str = "UNFUNDED") -> TotalReturnSwap:
    """A one-year TRS on ten million."""
    return TotalReturnSwap("TRS1", "IDX", "USD", "2024-01-01", "2025-01-01",
                           NOTIONAL, 50.0, "SOFR", "QUARTERLY", reset_type)


class TestFlatCurve:
    """A flat curve must be indistinguishable from the scalar it replaces."""

    def test_it_returns_its_rate_at_every_tenor(self):
        curve = RateCurve.flat(0.05)

        assert curve.zero_rate(0.0) == 0.05
        assert curve.zero_rate(0.25) == 0.05
        assert curve.zero_rate(30.0) == 0.05

    def test_it_reproduces_scalar_pricing_exactly(self):
        """The acceptance criterion — exact equality, not approximate.

        If a flat curve and a scalar rate disagreed even in the last bit, every
        existing derivatives result would shift the moment curves were adopted,
        and the diff would be impossible to distinguish from a real change.
        """
        curve = RateCurve.flat(0.05)

        for tenor in (0.25, 0.5, 1.0, 2.5, 10.0):
            scalar = cost_of_carry_fair_value(SPOT, 0.05, DIVIDEND_YIELD, tenor)
            from_curve = cost_of_carry_fair_value(
                SPOT, curve.zero_rate(tenor), DIVIDEND_YIELD, tenor)

            assert from_curve == scalar, f"disagreed at {tenor}y"

    def test_a_multi_pillar_curve_at_one_rate_is_also_exact(self):
        """Flatness is about the rates, not about having one pillar."""
        curve = RateCurve.from_pillars({0.5: 0.05, 2.0: 0.05, 10.0: 0.05})

        assert curve.is_flat
        assert curve.zero_rate(1.234) == 0.05

    def test_a_shaped_curve_is_not_flat(self):
        assert not RateCurve.from_pillars(PILLARS).is_flat


class TestInterpolation:

    @pytest.fixture
    def curve(self):
        return RateCurve.from_pillars(PILLARS)

    def test_it_returns_pillar_rates_exactly(self,
                                             curve):
        for tenor, rate in PILLARS.items():
            assert curve.zero_rate(tenor) == rate

    def test_it_interpolates_linearly_between_pillars(self,
                                                      curve):
        """Midway between 0.25y at 3% and 1y at 4% is 3.5%."""
        midpoint = 0.25 + (1.0 - 0.25) / 2

        assert curve.zero_rate(midpoint) == pytest.approx(0.035, abs=1e-12)

    def test_it_is_flat_before_the_first_pillar(self,
                                                curve):
        """Extrapolating a slope off the end is how a short rate becomes
        negative — arithmetically fine, financially nonsense, and the damage
        shows up far from the cause."""
        assert curve.zero_rate(0.01) == PILLARS[0.25]
        assert curve.zero_rate(0.0) == PILLARS[0.25]

    def test_it_is_flat_after_the_last_pillar(self,
                                              curve):
        assert curve.zero_rate(30.0) == PILLARS[5.0]

    def test_it_is_monotonic_on_a_rising_curve(self,
                                               curve):
        tenors = [0.1 * step for step in range(1, 60)]
        rates = [curve.zero_rate(tenor) for tenor in tenors]

        assert rates == sorted(rates)

    def test_the_pillars_round_trip_through_a_mapping(self):
        curve = RateCurve.from_pillars(PILLARS)

        assert curve.to_dict() == PILLARS
        assert RateCurve.from_pillars(curve.to_dict()) == curve

    def test_pillars_are_sorted_however_they_arrive(self):
        curve = RateCurve.from_pillars({5.0: 0.045, 0.25: 0.030, 1.0: 0.040})

        assert curve.tenors == (0.25, 1.0, 5.0)


class TestDiscountAndForward:

    @pytest.fixture
    def curve(self):
        return RateCurve.from_pillars(PILLARS)

    def test_a_zero_tenor_discounts_to_one(self,
                                           curve):
        assert curve.discount_factor(0.0) == 1.0

    def test_the_discount_factor_matches_the_definition(self,
                                                        curve):
        assert curve.discount_factor(1.0) == pytest.approx(
            math.exp(-0.040 * 1.0), rel=1e-15)

    def test_discount_factors_fall_with_tenor(self,
                                              curve):
        assert curve.discount_factor(5.0) < curve.discount_factor(1.0) < 1.0

    def test_the_forward_rate_matches_a_hand_calculation(self,
                                                         curve):
        """(0.045 x 5 - 0.040 x 1) / 4 = 0.04625."""
        assert curve.forward_rate(1.0, 5.0) == pytest.approx(0.04625, abs=1e-12)

    def test_the_forward_is_the_zero_rate_on_a_flat_curve(self):
        curve = RateCurve.flat(0.05)

        assert curve.forward_rate(1.0, 3.0) == pytest.approx(0.05, abs=1e-15)

    def test_compounding_forward_reproduces_the_discount_factor(self,
                                                                curve):
        """Discount to 1y then forward to 5y must equal discounting to 5y."""
        forward = curve.forward_rate(1.0, 5.0)
        stepwise = curve.discount_factor(1.0) * math.exp(-forward * 4.0)

        assert stepwise == pytest.approx(curve.discount_factor(5.0), rel=1e-12)

    def test_a_non_positive_period_is_refused(self,
                                              curve):
        with pytest.raises(CalculationError, match="forward period must be positive"):
            curve.forward_rate(2.0, 2.0)


class TestBumps:

    @pytest.fixture
    def curve(self):
        return RateCurve.from_pillars(PILLARS)

    def test_a_parallel_shift_moves_every_pillar(self,
                                                 curve):
        shifted = curve.shifted(BASIS_POINT)

        for tenor, rate in PILLARS.items():
            assert shifted.zero_rate(tenor) == pytest.approx(rate + BASIS_POINT,
                                                             abs=1e-15)

    def test_shifting_leaves_the_original_alone(self,
                                                curve):
        curve.shifted(0.01)

        assert curve.zero_rate(1.0) == PILLARS[1.0]

    def test_a_pillar_bump_moves_only_that_pillar(self,
                                                  curve):
        bumped = curve.with_pillar_bump(1.0, BASIS_POINT)

        assert bumped.zero_rate(1.0) == pytest.approx(0.040 + BASIS_POINT, abs=1e-15)
        assert bumped.zero_rate(0.25) == 0.030
        assert bumped.zero_rate(5.0) == 0.045

    def test_bumping_a_tenor_that_is_not_a_pillar_is_refused(self,
                                                             curve):
        """Adding a pillar would change the curve's shape, not its level, which
        is not what a key-rate bump means."""
        with pytest.raises(CalculationError, match="not a pillar"):
            curve.with_pillar_bump(2.0, BASIS_POINT)


class TestCurveValidation:

    def test_an_empty_curve_is_refused(self):
        with pytest.raises(CalculationError, match="at least one pillar"):
            RateCurve(tenors=(), rates=())

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(CalculationError, match="tenors but"):
            RateCurve(tenors=(1.0, 2.0), rates=(0.05,))

    def test_unsorted_tenors_are_refused(self):
        with pytest.raises(CalculationError, match="strictly increasing"):
            RateCurve(tenors=(2.0, 1.0), rates=(0.05, 0.04))

    def test_duplicate_tenors_are_refused(self):
        """Two rates at one tenor would make the interpolation divide by zero."""
        with pytest.raises(CalculationError, match="strictly increasing"):
            RateCurve(tenors=(1.0, 1.0), rates=(0.05, 0.04))

    def test_a_negative_tenor_is_refused(self):
        with pytest.raises(CalculationError, match="non-negative"):
            RateCurve(tenors=(-1.0,), rates=(0.05,))

    def test_asking_for_a_negative_tenor_is_refused(self):
        with pytest.raises(CalculationError, match="non-negative"):
            RateCurve.flat(0.05).zero_rate(-0.5)

    def test_from_pillars_refuses_an_empty_mapping(self):
        with pytest.raises(CalculationError, match="at least one pillar"):
            RateCurve.from_pillars({})


class TestSwapDV01:
    """The hand-computable case, and the sign that carries the information."""

    RESET = pd.Timestamp("2024-01-01")
    VALUATION = pd.Timestamp("2024-04-01")
    DAYS = 91

    def test_the_accrual_fraction_is_act_360(self):
        assert swap().financing_duration(self.VALUATION, self.RESET) == pytest.approx(
            self.DAYS / 360.0, rel=1e-15)

    def test_it_matches_the_hand_calculation(self):
        """notional x 1bp x 91/360 = 252.777..., negative to the receiver."""
        expected = -NOTIONAL * BASIS_POINT * (self.DAYS / 360.0)

        assert swap().dv01(self.VALUATION, self.RESET, 0.04) == pytest.approx(
            expected, rel=1e-12)

    def test_the_sign_is_negative_for_a_receiver(self):
        """The receiver pays financing, so a higher rate costs them.

        Reporting DV01 as a positive magnitude is common and loses exactly the
        thing a risk report needs: which way the position hurts.
        """
        assert swap().dv01(self.VALUATION, self.RESET, 0.04) < 0.0

    def test_it_scales_with_notional(self):
        small = TotalReturnSwap("TRS2", "IDX", "USD", "2024-01-01", "2025-01-01",
                                NOTIONAL / 10, 50.0, "SOFR", "QUARTERLY")

        assert small.dv01(self.VALUATION, self.RESET) == pytest.approx(
            swap().dv01(self.VALUATION, self.RESET) / 10, rel=1e-12)

    def test_it_scales_with_the_accrual_period(self):
        longer = swap().dv01(pd.Timestamp("2024-07-01"), self.RESET)
        shorter = swap().dv01(self.VALUATION, self.RESET)

        assert abs(longer) > abs(shorter)

    def test_it_does_not_depend_on_the_rate_level(self):
        """Financing is linear in the rate, so the slope is the same everywhere.

        Approximate rather than exact: the two differ in the last bit or two
        because the bump is added to a different-sized number, which is float
        noise and not a real dependence.
        """
        assert swap().dv01(self.VALUATION, self.RESET, 0.0) == pytest.approx(
            swap().dv01(self.VALUATION, self.RESET, 0.09), rel=1e-9)

    def test_a_funded_swap_has_no_rate_sensitivity(self):
        """Only the spread accrues, and the spread does not move with the rate."""
        assert swap("FUNDED").dv01(self.VALUATION, self.RESET, 0.04) == 0.0

    def test_bump_and_revalue_agrees_with_the_closed_form(self):
        """The reason the bumped version is the one kept."""
        contract = swap()
        base = contract.financing_cost(self.VALUATION, self.RESET, 0.04)
        bumped = contract.financing_cost(self.VALUATION, self.RESET,
                                         0.04 + BASIS_POINT)

        assert contract.dv01(self.VALUATION, self.RESET, 0.04) == pytest.approx(
            base - bumped, rel=1e-15)

    def test_a_valuation_before_the_reset_is_refused(self):
        with pytest.raises(ValueError, match="on or after"):
            swap().dv01(pd.Timestamp("2023-12-01"), self.RESET)


class TestTermStructure:

    @pytest.fixture
    def strip(self):
        return TermStructure(
            underlying="IDX",
            spot=SPOT,
            valuation_date=VALUATION_DATE,
            quotes=[FuturesQuote(pd.Timestamp("2024-03-15"), 100.9, "H4"),
                    FuturesQuote(pd.Timestamp("2024-06-21"), 101.9, "M4"),
                    FuturesQuote(pd.Timestamp("2024-09-20"), 103.5, "U4")],
            curve=RateCurve.from_pillars({0.25: 0.05, 1.0: 0.052}),
            dividend_yield=DIVIDEND_YIELD)

    def test_expiries_come_back_sorted(self):
        strip = TermStructure(
            underlying="IDX", spot=SPOT, valuation_date=VALUATION_DATE,
            quotes=[FuturesQuote(pd.Timestamp("2024-09-20")),
                    FuturesQuote(pd.Timestamp("2024-03-15"))],
            curve=RateCurve.flat(0.05))

        assert strip.expiries == [pd.Timestamp("2024-03-15"),
                                  pd.Timestamp("2024-09-20")]

    def test_theoretical_prices_rise_with_expiry(self,
                                                 strip):
        """Carry is positive here, so further out is dearer."""
        theoretical = strip.theoretical_prices()

        assert list(theoretical) == sorted(theoretical)

    def test_each_theoretical_price_matches_the_pricing_function(self,
                                                                 strip):
        """The container arranges; it does not reimplement."""
        theoretical = strip.theoretical_prices()

        for tenor, rate, value in zip(strip.times_to_expiry(),
                                      strip.financing_rates(),
                                      theoretical,
                                      strict=True):
            assert value == cost_of_carry_fair_value(SPOT, rate, DIVIDEND_YIELD, tenor)

    def test_basis_is_market_minus_theoretical(self,
                                               strip):
        expected = strip.market_prices() - strip.theoretical_prices()

        pd.testing.assert_series_equal(strip.basis(), expected, check_names=False)

    def test_basis_and_implied_repo_agree_on_direction(self,
                                                       strip):
        """Two views of one disagreement, so they cannot point opposite ways.

        A contract trading rich to the model must also imply a financing rate
        above the curve — if these ever disagreed, one of them would be wrong.
        """
        basis = strip.basis()
        implied = strip.implied_repo()
        financing = strip.financing_rates()

        for expiry, rate in zip(strip.expiries, financing, strict=True):
            assert (basis[expiry] > 0) == (implied[expiry] > rate), expiry

    def test_an_expiry_without_a_quote_reports_no_basis(self,
                                                        strip):
        """Inventing a basis for a contract nobody priced would be worse."""
        quiet = TermStructure(
            underlying="IDX", spot=SPOT, valuation_date=VALUATION_DATE,
            quotes=[FuturesQuote(pd.Timestamp("2024-06-21"))],
            curve=RateCurve.flat(0.05))

        assert quiet.basis().isna().all()
        assert quiet.implied_repo().isna().all()

    def test_the_frame_carries_every_column(self,
                                            strip):
        frame = strip.to_frame()

        assert list(frame.columns) == ["label", "time_to_expiry", "financing_rate",
                                       "theoretical", "market", "basis",
                                       "implied_repo"]
        assert len(frame) == 3

    def test_a_flat_curve_gives_one_financing_rate_everywhere(self):
        strip = TermStructure(
            underlying="IDX", spot=SPOT, valuation_date=VALUATION_DATE,
            quotes=[FuturesQuote(pd.Timestamp("2024-03-15")),
                    FuturesQuote(pd.Timestamp("2025-03-15"))],
            curve=RateCurve.flat(0.05))

        assert strip.financing_rates() == [0.05, 0.05]

    def test_no_expiries_is_refused(self):
        with pytest.raises(CalculationError, match="no expiries"):
            TermStructure(underlying="IDX", spot=SPOT,
                          valuation_date=VALUATION_DATE, quotes=[],
                          curve=RateCurve.flat(0.05))

    def test_a_non_positive_spot_is_refused(self):
        with pytest.raises(CalculationError, match="spot must be positive"):
            TermStructure(underlying="IDX", spot=0.0,
                          valuation_date=VALUATION_DATE,
                          quotes=[FuturesQuote(pd.Timestamp("2024-06-21"))],
                          curve=RateCurve.flat(0.05))

    def test_an_expired_contract_is_refused(self):
        with pytest.raises(CalculationError, match="before the valuation date"):
            TermStructure(underlying="IDX", spot=SPOT,
                          valuation_date=VALUATION_DATE,
                          quotes=[FuturesQuote(pd.Timestamp("2023-06-21"))],
                          curve=RateCurve.flat(0.05))


class TestSensitivityGrid:

    @pytest.fixture
    def grid(self):
        return sensitivity_grid(SPOT, GRID_TENORS, GRID_RATES,
                                dividend_yield=DIVIDEND_YIELD)

    def test_it_has_a_row_per_tenor_and_a_column_per_rate(self,
                                                          grid):
        assert grid.shape == (3, 3)
        assert list(grid.index) == GRID_TENORS
        assert list(grid.columns) == GRID_RATES

    def test_value_rises_with_the_rate(self,
                                       grid):
        """Higher financing means a dearer forward, at every tenor."""
        for tenor in GRID_TENORS:
            row = grid.loc[tenor]
            assert list(row) == sorted(row)

    def test_value_rises_with_tenor_when_carry_is_positive(self,
                                                           grid):
        for rate in GRID_RATES:
            column = grid[rate]
            assert list(column) == sorted(column)

    def test_each_cell_matches_the_pricing_function(self,
                                                    grid):
        for tenor in GRID_TENORS:
            for rate in GRID_RATES:
                assert grid.loc[tenor, rate] == cost_of_carry_fair_value(
                    SPOT, rate, DIVIDEND_YIELD, tenor)

    def test_the_axes_are_labelled(self,
                                   grid):
        assert grid.index.name == "time_to_expiry"
        assert grid.columns.name == "rate"

    def test_an_empty_axis_is_refused(self):
        with pytest.raises(CalculationError, match="non-empty"):
            sensitivity_grid(SPOT, [], GRID_RATES)
