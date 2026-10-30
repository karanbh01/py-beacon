# src/beacon/server/derivatives.py
"""
Stateless derivatives pricing.

Nothing here reads a stored document or writes one. A request carries every
input it needs and the response is a pure function of it, which is what makes
these endpoints safe to call repeatedly from a form as someone types, and what
lets a test assert that the storage directory is untouched afterwards.

The term-structure and roll reads are the exception only in that they resolve a
*price* from the data source. They still write nothing.

A futures fair value is ``S·e^((r − q + c)·T)``. The response splits the carry
into its financing, dividend and borrow components, each expressed as the
price effect it contributes rather than as a rate: "financing adds 1.24" is a
sentence about this contract, while "r is 5%" is a sentence about the world.
"""
# Where the maths lives: in `beacon.derivatives.pricing`, which has no Beacon
# imports at all and is checked against the textbook relationships directly,
# and since BN-417 in `beacon.derivatives.valuation`, which puts a curve's
# rates and a contract's dates into it. This module turns requests into those
# calls and their results into responses; it computes nothing.
import logging

import pandas as pd

from ..data.fetcher import DataFetcher
from ..derivatives.curves import RateCurve
from ..derivatives.swaps import TotalReturnSwap
from ..derivatives.term_structure import FuturesQuote, TermStructure
from ..derivatives.valuation import roll_between, tenor_of, value_future, value_swap
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


def price_futures(request: FuturesPriceRequest) -> FuturesPriceResponse:
    """Value a futures contract and decompose its carry.

    Args:
        request: Every input the calculation needs.

    Returns:
        FuturesPriceResponse: Fair value, the carry split into parts, contract
        value, and a tenor x rate grid.
    """
    found = value_future(spot=request.spot,
                         tenor=tenor_of(request.valuation_date, request.expiry,
                                        request.time_to_expiry),
                         curve=_curve(request.curve, request.risk_free_rate),
                         dividend_yield=request.dividend_yield,
                         borrow_cost=request.borrow_cost,
                         dividends=([(float(when), float(amount))
                                     for when, amount in request.dividends]
                                    if request.dividends else None),
                         market_price=request.market_price,
                         contract_multiplier=request.contract_multiplier,
                         contracts=request.contracts,
                         grid_tenors=request.grid_tenors,
                         grid_rates=request.grid_rates)
    carry = found.carry
    grid = found.sensitivity

    return FuturesPriceResponse(
        fair_value=found.fair_value,
        time_to_expiry=found.time_to_expiry,
        financing_rate=found.financing_rate,
        carry=CarryDecomposition(total=carry.total,
                                 financing=carry.financing,
                                 dividend=carry.dividend,
                                 borrow=carry.borrow,
                                 residual=carry.residual),
        contract_value=found.contract_value,
        market_price=request.market_price,
        basis=found.basis,
        implied_repo=found.implied_repo,
        sensitivity=TableFrame(index=[str(label) for label in grid.index],
                               columns=[str(label) for label in grid.columns],
                               data=[[float(value) for value in row]
                                     for row in grid.to_numpy()]))


def _curve(pillars: dict[str, float] | None,
           flat: float) -> RateCurve:
    """The financing curve: pillars if given, otherwise a flat rate.

    A flat curve reproduces scalar-rate pricing exactly, so a caller who sends
    one rate gets the same number they would have got before curves existed.
    """
    if pillars:
        return RateCurve.from_pillars({float(tenor): float(rate)
                                       for tenor, rate in pillars.items()})

    return RateCurve.flat(flat)


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
    found = value_swap(swap,
                       valuation,
                       pd.Timestamp(request.last_reset_date or request.start_date),
                       _curve(request.curve, request.reference_rate_value),
                       request.reference_rate_value,
                       request.spot,
                       request.initial_price,
                       dividends=request.dividends,
                       futures_prices=request.futures_prices,
                       time_to_expiry=request.time_to_expiry,
                       dividend_yield=request.dividend_yield)

    return TrsPriceResponse(
        trade_id=request.trade_id,
        valuation_date=str(valuation.date()),
        accrual_days=found.accrual_days,
        accrual_fraction=found.accrual_fraction,
        total_return_leg=found.total_return_leg,
        financing_leg=found.financing_leg,
        present_value=found.present_value,
        dv01=found.dv01,
        fair_spread_bps=found.fair_spread_bps,
        schedule=[TrsAccrual(start=str(period.start.date()),
                             end=str(period.end.date()),
                             days=period.days,
                             rate=period.rate,
                             accrual_fraction=period.accrual_fraction,
                             amount=period.amount)
                  for period in found.schedule],
        breakeven=found.breakeven)


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
    found = roll_between(spot,
                         valuation,
                         pd.Timestamp(front_expiry),
                         pd.Timestamp(back_expiry),
                         RateCurve.flat(risk_free_rate),
                         dividend_yield)

    return RollResponse(
        index_id=index_id,
        as_of=str(valuation.date()),
        spot=spot,
        front_expiry=str(pd.Timestamp(front_expiry).date()),
        back_expiry=str(pd.Timestamp(back_expiry).date()),
        front_price=found.front_price,
        back_price=found.back_price,
        roll_cost=found.roll_cost,
        annualised_roll=found.annualised_roll)


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
