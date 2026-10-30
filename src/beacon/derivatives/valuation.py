# src/beacon/derivatives/valuation.py
"""
A futures contract, a total return swap and a roll valued from their terms,
with everything shown beside the price: the carry split into its parts, a
sensitivity grid, a swap's financing schedule, its fair spread and its
breakeven against futures.

The formulas are `beacon.derivatives.pricing`'s; this module puts a curve's
rates and a contract's dates into them. Carry is shown as the *price effect*
of each rate, each priced with that rate alone. The parts do not sum to the
total exactly, because carry compounds rather than adds; the residual is
given rather than spread across them.
"""
# BN-417: moved from the server's derivatives views, which now call these.
import math
from dataclasses import dataclass, field

import pandas as pd

from ..exceptions import InvalidArgumentError
from .curves import RateCurve
from .pricing import (
    cost_of_carry_fair_value,
    discrete_dividend_fair_value,
    futures_roll_return,
    implied_repo_rate,
    trs_breakeven_spread,
)
from .swaps import TotalReturnSwap
from .term_structure import sensitivity_grid

# ACT/365, matching DerivativeBase.time_to_expiry.
DAYS_PER_YEAR = 365.0

# ACT/360, matching the financing leg in TotalReturnSwap.
FINANCING_DAYS_PER_YEAR = 360.0

BASIS_POINTS = 10_000.0

# The sensitivity grid's rates when none are given: the contract's own, and
# one and two points either side of it.
DEFAULT_RATE_SPREAD = (-0.02, -0.01, 0.0, 0.01, 0.02)

# Its tenors when none are given, as multiples of the contract's own.
DEFAULT_TENOR_MULTIPLES = (0.5, 1.0, 1.5, 2.0)

MONTHS_PER_PERIOD = {"MONTHLY": 1, "QUARTERLY": 3, "SEMI-ANNUAL": 6, "ANNUAL": 12}


def years_between(start: pd.Timestamp,
                  end: pd.Timestamp,
                  days_per_year: float = DAYS_PER_YEAR) -> float:
    """The year fraction from *start* to *end*, by calendar days."""
    return float((end - start).days / days_per_year)


def tenor_of(valuation_date: str | None,
             expiry: str | None,
             explicit: float | None) -> float:
    """Year fraction to expiry, ACT/365, from two dates or as given.

    Dates win when both are present: a tenor typed by hand can silently
    disagree with the expiry beside it.

    Raises:
        InvalidArgumentError: If the expiry is before the valuation date, no
            tenor can be found, or the tenor is negative.
    """
    if valuation_date is not None and expiry is not None:
        days = (pd.Timestamp(expiry) - pd.Timestamp(valuation_date)).days
        if days < 0:
            raise InvalidArgumentError(
                f"The expiry {expiry} is before the valuation date "
                f"{valuation_date}.")

        return float(days) / DAYS_PER_YEAR

    if explicit is None:
        raise InvalidArgumentError(
            "A time to expiry is needed: supply `time_to_expiry`, or both "
            "`valuation_date` and `expiry`.")

    if explicit < 0:
        raise InvalidArgumentError(
            f"The time to expiry must not be negative, got {explicit}.")

    return explicit


@dataclass(frozen=True)
class Carry:
    """A futures price's carry: its total, the price effect of each rate on
    its own, and what compounding leaves between them."""
    total: float
    financing: float
    dividend: float
    borrow: float
    residual: float


def carry_of(spot: float,
             rate: float,
             tenor: float,
             dividend_yield: float,
             borrow_cost: float,
             fair_value: float) -> Carry:
    """The carry from *spot* to *fair_value*, split by rate."""
    financing = spot * math.expm1(rate * tenor)
    dividend = -spot * math.expm1(dividend_yield * tenor)
    borrow = spot * math.expm1(borrow_cost * tenor)
    total = fair_value - spot

    return Carry(total=total,
                 financing=financing,
                 dividend=dividend,
                 borrow=borrow,
                 residual=total - (financing + dividend + borrow))


@dataclass(frozen=True)
class FuturesValuation:
    """A futures contract valued from its terms.

    Attributes:
        fair_value: Its theoretical price.
        time_to_expiry: In years.
        financing_rate: The curve's rate to expiry.
        carry: The carry, split by rate.
        contract_value: The fair value times the multiplier and contracts.
        basis: The market price less the fair value, when quoted.
        implied_repo: The financing rate the market price implies.
        sensitivity: Fair value across tenors (rows) and rates (columns).
    """
    fair_value: float
    time_to_expiry: float
    financing_rate: float
    carry: Carry
    contract_value: float
    basis: float | None
    implied_repo: float | None
    sensitivity: pd.DataFrame


def value_future(spot: float,
                 tenor: float,
                 curve: RateCurve,
                 dividend_yield: float = 0.0,
                 borrow_cost: float = 0.0,
                 dividends: list[tuple[float, float]] | None = None,
                 market_price: float | None = None,
                 contract_multiplier: float = 1.0,
                 contracts: float = 1.0,
                 grid_tenors: list[float] | None = None,
                 grid_rates: list[float] | None = None) -> FuturesValuation:
    """A futures contract on *spot*, *tenor* years out, financed on *curve*.

    Args:
        spot: The underlying's price.
        tenor: Years to expiry.
        curve: The financing curve.
        dividend_yield: A continuous yield, used without *dividends*.
        borrow_cost: The cost of borrowing the underlying.
        dividends: Discrete dividends as (years from now, amount); priced
            off them instead of the yield when given.
        market_price: A quote to measure the basis and implied repo from.
        contract_multiplier: Value per point.
        contracts: How many.
        grid_tenors: The sensitivity grid's tenors; multiples of *tenor*
            when None.
        grid_rates: Its rates; spreads around the curve's when None.
    """
    rate = curve.zero_rate(tenor)

    if dividends:
        fair_value = discrete_dividend_fair_value(spot=spot,
                                                  risk_free_rate=rate,
                                                  time_to_expiry_years=tenor,
                                                  dividends=dividends)
    else:
        fair_value = cost_of_carry_fair_value(spot=spot,
                                              risk_free_rate=rate,
                                              dividend_yield=dividend_yield,
                                              time_to_expiry_years=tenor,
                                              borrow_cost=borrow_cost)

    implied = (implied_repo_rate(futures_price=market_price,
                                 spot=spot,
                                 dividend_yield=dividend_yield,
                                 time_to_expiry_years=tenor)
               if market_price is not None and tenor > 0.0 else None)
    tenors = grid_tenors or [tenor * multiple for multiple in DEFAULT_TENOR_MULTIPLES]
    rates = grid_rates or [rate + spread for spread in DEFAULT_RATE_SPREAD]

    return FuturesValuation(
        fair_value=fair_value,
        time_to_expiry=tenor,
        financing_rate=rate,
        carry=carry_of(spot, rate, tenor, dividend_yield, borrow_cost, fair_value),
        contract_value=fair_value * contract_multiplier * contracts,
        basis=market_price - fair_value if market_price is not None else None,
        implied_repo=implied,
        sensitivity=sensitivity_grid(spot=spot,
                                     tenors=list(tenors),
                                     rates=list(rates),
                                     dividend_yield=dividend_yield,
                                     borrow_cost=borrow_cost))


@dataclass(frozen=True)
class Accrual:
    """One financing period of a swap."""
    start: pd.Timestamp
    end: pd.Timestamp
    days: int
    rate: float
    accrual_fraction: float
    amount: float


@dataclass(frozen=True)
class SwapValuation:
    """A total return swap valued at a date.

    Attributes:
        accrual_days: Days since the last reset.
        accrual_fraction: Their ACT/360 year fraction.
        total_return_leg: What the underlying returned on the notional.
        financing_leg: The financing accrued since the last reset.
        present_value: The first less the second.
        dv01: The financing leg's change for a basis point.
        fair_spread_bps: The spread at which the swap is worth nothing
            today; None when nothing has accrued.
        schedule: Each financing period from the last reset to maturity.
        breakeven: The spread at which the swap matches each futures price.
    """
    accrual_days: int
    accrual_fraction: float
    total_return_leg: float
    financing_leg: float
    present_value: float
    dv01: float
    fair_spread_bps: float | None
    schedule: list[Accrual]
    breakeven: list[dict[str, float]] = field(default_factory=list)


def value_swap(swap: TotalReturnSwap,
               valuation: pd.Timestamp,
               last_reset: pd.Timestamp,
               curve: RateCurve,
               fixed_rate: float,
               spot: float,
               initial_price: float,
               dividends: float = 0.0,
               futures_prices: list[float] | None = None,
               time_to_expiry: float | None = None,
               dividend_yield: float = 0.0) -> SwapValuation:
    """*swap* valued at *valuation*, its current period fixed at
    *fixed_rate* since *last_reset* and the later ones projected off
    *curve*."""
    accrued = swap.financing_cost(valuation, last_reset, fixed_rate)
    total_return_leg = swap.total_return_leg(spot, initial_price, dividends)
    fraction = swap.financing_duration(valuation, last_reset)

    return SwapValuation(
        accrual_days=int((valuation - last_reset).days),
        accrual_fraction=fraction,
        total_return_leg=total_return_leg,
        financing_leg=accrued,
        present_value=total_return_leg - accrued,
        dv01=swap.dv01(valuation, last_reset, fixed_rate),
        fair_spread_bps=(None if fraction <= 0.0 else
                         (total_return_leg / (swap.notional * fraction) - fixed_rate)
                         * BASIS_POINTS),
        schedule=financing_schedule(swap, valuation, last_reset, curve, fixed_rate),
        breakeven=([] if futures_prices is None or time_to_expiry is None else
                   [{"futures_price": float(price),
                     "breakeven_spread_bps": trs_breakeven_spread(
                         futures_price=float(price),
                         spot=spot,
                         risk_free_rate=fixed_rate,
                         time_to_expiry_years=time_to_expiry,
                         dividend_yield=dividend_yield) * BASIS_POINTS}
                    for price in futures_prices]))


def financing_schedule(swap: TotalReturnSwap,
                       valuation: pd.Timestamp,
                       last_reset: pd.Timestamp,
                       curve: RateCurve,
                       fixed_rate: float) -> list[Accrual]:
    """Each financing period from the last reset to maturity.

    The current period accrues at the rate fixed at its reset; each later one
    at the curve's forward rate over it, so a flat curve reproduces the fixed
    rate and a shaped one prices later periods off the later curve.
    """
    end = pd.Timestamp(swap.end_date)
    step = pd.DateOffset(months=MONTHS_PER_PERIOD[swap.payment_frequency.upper()])
    rows: list[Accrual] = []
    period_start = last_reset

    while period_start < end:
        period_end = min(period_start + step, end)
        days = int((period_end - period_start).days)
        if days <= 0:
            break

        rate = _period_rate(curve, valuation, period_start, period_end, fixed_rate)
        fraction = days / FINANCING_DAYS_PER_YEAR
        rows.append(Accrual(start=period_start,
                            end=period_end,
                            days=days,
                            rate=rate,
                            accrual_fraction=fraction,
                            amount=swap.notional * (rate + swap.spread) * fraction))
        period_start = period_end

    return rows


def _period_rate(curve: RateCurve,
                 valuation: pd.Timestamp,
                 period_start: pd.Timestamp,
                 period_end: pd.Timestamp,
                 fixed: float) -> float:
    """The rate a period accrues at: the fixed one for the current period,
    the curve's forward over it for a later one."""
    if period_start <= valuation:
        return fixed

    start_years = years_between(valuation, period_start)
    end_years = years_between(valuation, period_end)

    if end_years - start_years <= 0.0:
        return fixed

    return curve.forward_rate(start_years, end_years)


@dataclass(frozen=True)
class Roll:
    """Rolling from one contract to the next, both priced off the same spot
    and curve: the carry roll rather than a market one."""
    front_price: float
    back_price: float
    roll_cost: float
    annualised_roll: float


def roll_between(spot: float,
                 valuation: pd.Timestamp,
                 front_expiry: pd.Timestamp,
                 back_expiry: pd.Timestamp,
                 curve: RateCurve,
                 dividend_yield: float = 0.0) -> Roll:
    """The roll from the *front_expiry* contract to the *back_expiry* one.

    Raises:
        InvalidArgumentError: If the back expiry is not after the front.
    """
    front_tenor = years_between(valuation, front_expiry)
    back_tenor = years_between(valuation, back_expiry)

    if back_tenor <= front_tenor:
        raise InvalidArgumentError(
            f"The back expiry {back_expiry.date()} must be after the front expiry "
            f"{front_expiry.date()}.")

    front = cost_of_carry_fair_value(spot, curve.zero_rate(front_tenor),
                                     dividend_yield, front_tenor)
    back = cost_of_carry_fair_value(spot, curve.zero_rate(back_tenor),
                                    dividend_yield, back_tenor)

    return Roll(front_price=front,
                back_price=back,
                roll_cost=back - front,
                annualised_roll=futures_roll_return(front, back, front_expiry, back_expiry))
