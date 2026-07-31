# src/beacon/server/derivatives.py
"""
Stateless derivatives pricing.

Nothing here reads a stored document or writes one. A request carries every
input it needs and the response is a pure function of it, which is what makes
these endpoints safe to call repeatedly from a form as someone types — and what
lets a test assert that the storage directory is untouched afterwards.

The term-structure and roll reads are the exception only in that they resolve a
*price* from the data source. They still write nothing.

## Where the maths lives

In `beacon.derivatives.pricing`, which has no Beacon imports at all and is
checked against the textbook relationships directly. This module arranges
inputs and shapes outputs; it does not re-derive anything. That is the same
boundary BN-96 drew for curves: a curve is asked for a rate, the rate goes into
a pricing function, and nothing about either crosses into the other.

## Carry, decomposed

A futures fair value is ``S·e^((r − q + c)·T)``, and the interesting part for
someone looking at a screen is not the total but which piece of it is which.
The response therefore splits the carry into its financing, dividend and borrow
components, each expressed as the price effect it contributes rather than as a
rate — because "financing adds 1.24" is a sentence about this contract, while
"r is 5%" is a sentence about the world.
"""
import logging
import math

import pandas as pd

from ..data.fetcher import DataFetcher
from ..derivatives.curves import RateCurve
from ..derivatives.pricing import (
    cost_of_carry_fair_value,
    discrete_dividend_fair_value,
    futures_roll_return,
    implied_repo_rate,
    trs_breakeven_spread,
)
from ..derivatives.swaps import TotalReturnSwap
from ..derivatives.term_structure import FuturesQuote, TermStructure, sensitivity_grid
from ..exceptions import DataNotFoundError
from .schemas import (
    CarryDecomposition,
    FuturesPriceRequest,
    FuturesPriceResponse,
    RollResponse,
    TableFrame,
    TermStructureEntry,
    TermStructureResponse,
    TrsAccrual,
    TrsPriceRequest,
    TrsPriceResponse,
)

logger = logging.getLogger(__name__)

# ACT/365, matching DerivativeBase.time_to_expiry.
DAYS_PER_YEAR = 365.0

# ACT/360, matching the financing leg in TotalReturnSwap.
FINANCING_DAYS_PER_YEAR = 360.0

# Rows and columns of the sensitivity grid when a caller does not supply axes.
DEFAULT_TENORS = (0.25, 0.5, 0.75, 1.0)
DEFAULT_RATE_SPREAD = (-0.02, -0.01, 0.0, 0.01, 0.02)


def price_futures(request: FuturesPriceRequest) -> FuturesPriceResponse:
    """Value a futures contract and decompose its carry.

    Args:
        request: Every input the calculation needs.

    Returns:
        FuturesPriceResponse: Fair value, the carry split into parts, contract
        value, and a tenor x rate grid.
    """
    tenor = _tenor(request.valuation_date, request.expiry, request.time_to_expiry)
    curve = _curve(request)
    rate = curve.zero_rate(tenor)

    fair_value = _fair_value(request, rate, tenor)
    basis = (request.market_price - fair_value
             if request.market_price is not None else None)

    return FuturesPriceResponse(
        fair_value=fair_value,
        time_to_expiry=tenor,
        financing_rate=rate,
        carry=_carry(request, rate, tenor, fair_value),
        contract_value=fair_value * request.contract_multiplier * request.contracts,
        market_price=request.market_price,
        basis=basis,
        implied_repo=_implied_repo(request, tenor),
        sensitivity=_grid(request, rate, tenor))


def _tenor(valuation_date: str | None,
           expiry: str | None,
           explicit: float | None) -> float:
    """Year fraction to expiry, ACT/365.

    A caller may state the tenor directly or give two dates. Dates win when
    both are present, because they are the less ambiguous statement — a tenor
    typed by hand can silently disagree with the expiry beside it.
    """
    if valuation_date is not None and expiry is not None:
        days = (pd.Timestamp(expiry) - pd.Timestamp(valuation_date)).days
        if days < 0:
            raise DataNotFoundError(
                f"a future expiry: {expiry} precedes {valuation_date}",
                source="futures pricing")

        return float(days) / DAYS_PER_YEAR

    if explicit is None:
        raise DataNotFoundError(
            "a time to expiry",
            source="supply `time_to_expiry`, or both `valuation_date` and "
                   "`expiry`")

    if explicit < 0:
        raise DataNotFoundError(
            f"a non-negative time to expiry, got {explicit}",
            source="futures pricing")

    return explicit


def _curve(request: FuturesPriceRequest) -> RateCurve:
    """The financing curve: pillars if given, otherwise a flat rate.

    A flat curve reproduces scalar-rate pricing exactly, so a caller who sends
    one rate gets the same number they would have got before curves existed.
    """
    if request.curve:
        return RateCurve.from_pillars(
            {float(tenor): float(rate) for tenor, rate in request.curve.items()})

    return RateCurve.flat(request.risk_free_rate)


def _fair_value(request: FuturesPriceRequest,
                rate: float,
                tenor: float) -> float:
    """Theoretical price, by whichever dividend treatment was asked for."""
    if request.dividends:
        return discrete_dividend_fair_value(
            spot=request.spot,
            risk_free_rate=rate,
            time_to_expiry_years=tenor,
            dividends=[(float(when), float(amount))
                       for when, amount in request.dividends])

    return cost_of_carry_fair_value(spot=request.spot,
                                    risk_free_rate=rate,
                                    dividend_yield=request.dividend_yield,
                                    time_to_expiry_years=tenor,
                                    borrow_cost=request.borrow_cost)


def _carry(request: FuturesPriceRequest,
           rate: float,
           tenor: float,
           fair_value: float) -> CarryDecomposition:
    """Split the carry into the pieces a person can reason about.

    Each component is the *price effect* of one rate, computed by pricing with
    that rate alone. They do not sum to the total exactly, because carry
    compounds rather than adds — the residual is reported rather than spread
    across the parts, which would make each of them slightly wrong to hide the
    fact that the decomposition is approximate.
    """
    spot = request.spot

    financing = spot * math.expm1(rate * tenor)
    dividend = -spot * math.expm1(request.dividend_yield * tenor)
    borrow = spot * math.expm1(request.borrow_cost * tenor)

    total = fair_value - spot

    return CarryDecomposition(
        total=total,
        financing=financing,
        dividend=dividend,
        borrow=borrow,
        residual=total - (financing + dividend + borrow))


def _implied_repo(request: FuturesPriceRequest,
                  tenor: float) -> float | None:
    """The financing rate a quoted price implies, when there is a quote."""
    if request.market_price is None or tenor <= 0.0:
        return None

    return implied_repo_rate(futures_price=request.market_price,
                             spot=request.spot,
                             dividend_yield=request.dividend_yield,
                             time_to_expiry_years=tenor)


def _grid(request: FuturesPriceRequest,
          rate: float,
          tenor: float) -> TableFrame:
    """Fair value across a tenor x rate grid, centred on this contract."""
    tenors = request.grid_tenors or [tenor * multiple
                                     for multiple in (0.5, 1.0, 1.5, 2.0)]
    rates = request.grid_rates or [rate + spread
                                   for spread in DEFAULT_RATE_SPREAD]

    frame = sensitivity_grid(spot=request.spot,
                             tenors=list(tenors),
                             rates=list(rates),
                             dividend_yield=request.dividend_yield,
                             borrow_cost=request.borrow_cost)

    return TableFrame(index=[str(label) for label in frame.index],
                      columns=[str(label) for label in frame.columns],
                      data=[[float(value) for value in row]
                            for row in frame.to_numpy()])


def price_trs(request: TrsPriceRequest) -> TrsPriceResponse:
    """Value a total return swap and schedule its financing.

    Args:
        request: Trade terms, legs and valuation inputs.

    Returns:
        TrsPriceResponse: Financing schedule, present value, fair spread,
        breakeven table and DV01.
    """
    swap = TotalReturnSwap(derivative_id=request.trade_id,
                           underlying_id=request.underlying_id,
                           currency=request.currency,
                           start_date=request.start_date,
                           end_date=request.end_date,
                           notional=request.notional,
                           spread_bps=request.spread_bps,
                           reference_rate=request.reference_rate,
                           payment_frequency=request.payment_frequency,
                           reset_type=request.reset_type)

    valuation = pd.Timestamp(request.valuation_date)
    last_reset = pd.Timestamp(request.last_reset_date or request.start_date)

    curve = (RateCurve.from_pillars({float(t): float(r)
                                     for t, r in request.curve.items()})
             if request.curve else RateCurve.flat(request.reference_rate_value))

    schedule = _financing_schedule(swap, request, curve)
    accrued = swap.financing_cost(valuation, last_reset,
                                  request.reference_rate_value)

    total_return_leg = request.notional * (request.spot / request.initial_price - 1.0)

    return TrsPriceResponse(
        trade_id=request.trade_id,
        valuation_date=str(valuation.date()),
        accrual_days=int((valuation - last_reset).days),
        accrual_fraction=swap.financing_duration(valuation, last_reset),
        total_return_leg=total_return_leg,
        financing_leg=accrued,
        present_value=total_return_leg - accrued,
        dv01=swap.dv01(valuation, last_reset, request.reference_rate_value),
        fair_spread_bps=_fair_spread_bps(request, total_return_leg, valuation,
                                         last_reset, swap),
        schedule=schedule,
        breakeven=_breakeven(request))


def _financing_schedule(swap: TotalReturnSwap,
                        request: TrsPriceRequest,
                        curve: RateCurve) -> list[TrsAccrual]:
    """Accruals for each period from the last reset to maturity.

    Each future period is projected at the curve's forward rate over that
    period, which is what the curve is for: a flat curve reproduces the scalar
    rate exactly, and a shaped one prices the later periods off the later part
    of the curve rather than off today's fixing.
    """
    start = pd.Timestamp(request.last_reset_date or request.start_date)
    end = pd.Timestamp(request.end_date)
    step = _period_offset(request.payment_frequency)

    rows: list[TrsAccrual] = []
    period_start = start
    valuation = pd.Timestamp(request.valuation_date)

    while period_start < end:
        period_end = min(period_start + step, end)
        days = int((period_end - period_start).days)
        if days <= 0:
            break

        rate = _period_rate(curve, valuation, period_start, period_end,
                            request.reference_rate_value)
        fraction = days / FINANCING_DAYS_PER_YEAR
        accrual = swap.notional * (rate + swap.spread) * fraction

        rows.append(TrsAccrual(start=str(period_start.date()),
                               end=str(period_end.date()),
                               days=days,
                               rate=rate,
                               accrual_fraction=fraction,
                               amount=accrual))
        period_start = period_end

    return rows


def _period_rate(curve: RateCurve,
                 valuation: pd.Timestamp,
                 period_start: pd.Timestamp,
                 period_end: pd.Timestamp,
                 fixed: float) -> float:
    """The rate a period accrues at.

    The current period uses the rate already fixed at its reset — that is what
    "fixed" means, and projecting it off a curve would overwrite an observed
    number with a modelled one. Later periods are projected forward.
    """
    if period_start <= valuation:
        return fixed

    start_years = (period_start - valuation).days / DAYS_PER_YEAR
    end_years = (period_end - valuation).days / DAYS_PER_YEAR

    if end_years - start_years <= 0.0:
        return fixed

    return curve.forward_rate(start_years, end_years)


def _period_offset(frequency: str) -> pd.DateOffset:
    """Months per payment period."""
    months = {"MONTHLY": 1, "QUARTERLY": 3, "SEMI-ANNUAL": 6, "ANNUAL": 12}

    return pd.DateOffset(months=months[frequency.upper()])


def _fair_spread_bps(request: TrsPriceRequest,
                     total_return_leg: float,
                     valuation: pd.Timestamp,
                     last_reset: pd.Timestamp,
                     swap: TotalReturnSwap) -> float | None:
    """The spread at which this swap would be worth nothing today.

    None when no time has accrued: with a zero day-count fraction the financing
    leg cannot be moved by any spread, so there is no level that would balance
    the trade rather than an infinitely large one.
    """
    fraction = swap.financing_duration(valuation, last_reset)
    if fraction <= 0.0:
        return None

    required_rate = total_return_leg / (request.notional * fraction)

    return (required_rate - request.reference_rate_value) * 10_000.0


def _breakeven(request: TrsPriceRequest) -> list[dict[str, float]]:
    """Breakeven financing spread against a range of futures prices.

    The comparison a desk actually makes: a TRS and a future are two ways to
    hold the same exposure, so the question is what spread makes them agree.
    """
    if request.futures_prices is None or request.time_to_expiry is None:
        return []

    return [
        {"futures_price": float(price),
         "breakeven_spread_bps": trs_breakeven_spread(
             futures_price=float(price),
             spot=request.spot,
             risk_free_rate=request.reference_rate_value,
             time_to_expiry_years=request.time_to_expiry,
             dividend_yield=request.dividend_yield) * 10_000.0}
        for price in request.futures_prices
    ]


def build_term_structure(index_id: str,
                         fetcher: DataFetcher,
                         expiries: list[str],
                         as_of: str | None,
                         risk_free_rate: float,
                         dividend_yield: float) -> TermStructureResponse:
    """Price a strip of futures on an index, off its own spot.

    Raises:
        DataNotFoundError: If the index cannot be priced.
    """
    spot, valuation = _spot_of(index_id, fetcher, as_of)

    strip = TermStructure(underlying=index_id,
                          spot=spot,
                          valuation_date=valuation,
                          quotes=[FuturesQuote(pd.Timestamp(expiry))
                                  for expiry in expiries],
                          curve=RateCurve.flat(risk_free_rate),
                          dividend_yield=dividend_yield)

    frame = strip.to_frame()

    return TermStructureResponse(
        index_id=index_id,
        as_of=str(valuation.date()),
        spot=spot,
        entries=[TermStructureEntry(
            expiry=str(pd.Timestamp(expiry).date()),
            time_to_expiry=float(row["time_to_expiry"]),
            financing_rate=float(row["financing_rate"]),
            theoretical=float(row["theoretical"]))
            for expiry, row in frame.iterrows()])


def build_roll(index_id: str,
               fetcher: DataFetcher,
               front_expiry: str,
               back_expiry: str,
               as_of: str | None,
               risk_free_rate: float,
               dividend_yield: float) -> RollResponse:
    """The cost or gain of rolling from one contract to the next.

    Both legs are priced theoretically off the same spot and curve, so the roll
    reported here is the *carry* roll rather than a market one. With a flat
    curve it is positive in backwardation and negative in contango, which is
    the sign convention a desk expects.
    """
    spot, valuation = _spot_of(index_id, fetcher, as_of)
    curve = RateCurve.flat(risk_free_rate)

    front_tenor = (pd.Timestamp(front_expiry) - valuation).days / DAYS_PER_YEAR
    back_tenor = (pd.Timestamp(back_expiry) - valuation).days / DAYS_PER_YEAR

    if back_tenor <= front_tenor:
        raise DataNotFoundError(
            f"a back expiry after the front: {back_expiry} does not follow "
            f"{front_expiry}",
            source="roll pricing")

    front = cost_of_carry_fair_value(spot, curve.zero_rate(front_tenor),
                                     dividend_yield, front_tenor)
    back = cost_of_carry_fair_value(spot, curve.zero_rate(back_tenor),
                                    dividend_yield, back_tenor)

    return RollResponse(
        index_id=index_id,
        as_of=str(valuation.date()),
        spot=spot,
        front_expiry=str(pd.Timestamp(front_expiry).date()),
        back_expiry=str(pd.Timestamp(back_expiry).date()),
        front_price=front,
        back_price=back,
        roll_cost=back - front,
        annualised_roll=futures_roll_return(front, back,
                                            pd.Timestamp(front_expiry),
                                            pd.Timestamp(back_expiry)))


def _spot_of(index_id: str,
             fetcher: DataFetcher,
             as_of: str | None) -> tuple[float, pd.Timestamp]:
    """The index's close on or before a date, and that date."""
    frame = fetcher.fetch_market_data(index_id, None, as_of)
    if frame.empty or "CLOSE" not in frame.columns:
        raise DataNotFoundError(f"a price for '{index_id}'", source="MarketData")

    closes = frame["CLOSE"].dropna()
    if closes.empty:
        raise DataNotFoundError(f"a price for '{index_id}'", source="MarketData")

    return float(closes.iloc[-1]), pd.Timestamp(closes.index[-1])
