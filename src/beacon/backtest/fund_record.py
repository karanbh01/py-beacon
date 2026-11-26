# src/beacon/backtest/fund_record.py
"""
The vehicle's recorded data: `result.fund`.

    result.fund.nav_per_unit      a unit's value each day
    result.fund.trading_nav       the fund's net assets each simulated day
    result.fund.fees.payable      the fee owed and not yet paid each day
    result.fund.dividends.entitled
    result.fund.swap.value        a synthetic fund's swap, else None

The fund's net assets are the portfolio's NAV less the management fee owed
and not yet paid, plus the dividends owed to it, a synthetic fund's swap and
hedged share classes' forwards, as its accounts would show them. The
vehicle's parts (``etf``, ``swap``, ``futures``, ``cash_band``, ``hedging``)
are None when the run's vehicle lacks them, not empty.

Recorded data only: what was measured from it is under `result.analytics`.
"""
# BN-459 (Karan, 2026-11-24): the vehicle's record in one place; the result's
# flat names warn until 0.12.0. The NAV identities moved here from
# backtest.metrics, which reads them from it.
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from ..portfolio.base import Portfolio
from ..portfolio.cash_flows import FEE
from .flows import FlowRecord
from .futures import FuturesRoll, FuturesTrade
from .share_classes import ShareClassBook
from .swap import SwapReset

Nav = Callable[[], pd.Series]


@dataclass
class FeesRecord:
    """The management fee.

    Attributes:
        payable: Owed and not yet paid at the end of each day, which the NAV
            is net of.
        paid: Paid out of the fund over the run.
    """
    payable: pd.Series
    paid: float


@dataclass
class DividendsRecord:
    """The dividends the fund's holdings paid.

    Attributes:
        receivable: Owed at the end of each day: gone ex and not yet paid,
            at the day's exchange rates and net of withholding tax. The NAV
            includes it, as a fund's accounts do; nothing is bought with it
            until it is paid.
        entitled: What the fund became owed on each day a holding went ex:
            ``GROSS``, before tax, and ``WITHHELD``, the tax withheld, at
            that day's exchange rates. Only days with a dividend have a row.
    """
    receivable: pd.Series
    entitled: pd.DataFrame


@dataclass
class ETFRecord:
    """An ETF's life on its exchange.

    Attributes:
        market: Each day's quote: NAV per share, premium, spread, market
            price, bid and ask, and the arbitrage band.
    """
    market: pd.DataFrame


@dataclass
class SwapRecord:
    """A synthetic fund's total return swap.

    Attributes:
        value: Its value at the end of each day: owed to the fund when
            positive, by it when negative. The NAV includes it.
        resets: Each reset: the day, the value settled, why, and the notional
            it was reset to.
    """
    value: pd.Series
    resets: list[SwapReset]
    _nav: Nav = field(repr=False, compare=False)

    @property
    def counterparty_exposure(self) -> pd.Series:
        """What the fund would lose if its counterparty failed, as a share
        of NAV at the end of each day: the swap's value when owed to the
        fund, otherwise zero."""
        nav = self._nav()

        return (self.value.clip(lower=0.0).reindex(nav.index).fillna(0.0)
                / nav).rename("counterparty_exposure")


@dataclass
class FuturesRecord:
    """The index futures a fund held, futures-based or equitising its cash.

    Attributes:
        notional: The exposure they carried at the end of each day. Their
            profit and loss is already in the cash.
        trades: Each trade: opened, rolled or resized, the contracts before,
            after and traded, the price, the notional traded and the cost.
        rolls: Each roll to the next contract: the day, the new expiry, the
            notional, the cost and the dividend yield it was priced with.
        days: Each day they were held: the contracts, their price, their
            fair value, the level of the index they are on and of the fund's
            own, the assumed mispricing and their expiry.
    """
    notional: pd.Series
    trades: list[FuturesTrade]
    rolls: list[FuturesRoll]
    days: pd.DataFrame
    _nav: Nav = field(repr=False, compare=False)

    @property
    def exposure(self) -> pd.Series:
        """The exposure they carried as a share of NAV at the end of each
        day."""
        nav = self._nav()

        return (self.notional.reindex(nav.index).fillna(0.0) / nav).rename("futures_exposure")


@dataclass
class CashBandRecord:
    """A fund's cash band.

    Attributes:
        events: Each time its cash left the band and was brought back, by
            date: a raise or a spend-down, the cash's share of the fund
            before and after, the target and the band, the value traded, its
            cost and the number of trades.
    """
    events: pd.DataFrame


@dataclass
class HedgingRecord:
    """Currency-hedged share classes' forwards.

    Attributes:
        value: The value of their open forwards at the end of each day, in
            the fund's currency. The NAV includes it.
    """
    value: pd.Series


@dataclass
class FundRecord:
    """The vehicle's recorded data.

    Attributes:
        currency: The fund's currency, which its NAV, costs and holdings'
            values are in.
        launch_price: NAV per unit at launch.
        units: Units outstanding at the end of each day, from the eve of the
            first trading day.
        flows: Each day's flow as dealt: the amount, and the units created
            or cancelled at that day's NAV per unit. Empty without flows.
        fees: The management fee owed each day and paid over the run.
        dividends: The dividends the fund was owed.
        share_classes: For a fund run with share classes, each class's
            record by name: its net assets, units, NAV per unit, fees,
            distributions and flows. Empty otherwise.
        etf: An ETF's quotes on its exchange; None for another vehicle.
        swap: A synthetic fund's swap; None otherwise.
        futures: The index futures held; None when none were.
        cash_band: A cash band's events; None for a fund without one.
        hedging: Hedged share classes' forwards; None without them.
    """
    portfolio: Portfolio = field(repr=False)
    currency: str = "USD"
    launch_price: float = 1.0
    units: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    flows: list[FlowRecord] = field(default_factory=list)
    fees: FeesRecord = field(default_factory=lambda: FeesRecord(pd.Series(dtype=float), 0.0))
    dividends: DividendsRecord = field(default_factory=lambda: DividendsRecord(
        pd.Series(dtype=float), pd.DataFrame(columns=["GROSS", "WITHHELD"], dtype=float)))
    share_classes: dict[str, ShareClassBook] = field(default_factory=dict)
    etf: ETFRecord | None = None
    swap: SwapRecord | None = None
    futures: FuturesRecord | None = None
    cash_band: CashBandRecord | None = None
    hedging: HedgingRecord | None = None

    @property
    def trading_nav(self) -> pd.Series:
        """Net assets over the simulated days, with the day-zero row
        excluded.

        The portfolio's own `nav` opens with initial capital on the eve of
        the first trading day, the record of what the run started with.
        Every *metric* derives from this series instead: the eve row is a
        starting fact, not a day the simulation traded.
        """
        nav = self.net_assets()

        if (self.portfolio.inception is not None and not nav.empty
                and nav.index[0] == self.portfolio.inception):
            return nav.iloc[1:]

        return nav

    @property
    def aum(self) -> pd.Series:
        """Assets under management each simulated day: the trading NAV,
        which flows move as well as the market."""
        return self.trading_nav

    @property
    def units_outstanding(self) -> pd.Series:
        """Units outstanding at the end of each simulated day."""
        return self.units_on(self.trading_nav.index)

    @property
    def nav_per_unit(self) -> pd.Series:
        """NAV per unit each simulated day: the performance of a unit held
        throughout, whatever money arrived or left. A day with no units
        outstanding carries the last NAV per unit."""
        nav = self.trading_nav
        units = self.units_on(nav.index)
        per_unit = (nav / units).where(units > 0.0)

        return per_unit.ffill().fillna(self.launch_price)

    def net_assets(self) -> pd.Series:
        """The portfolio's NAV less the fee owed each day, plus the
        dividends owed, a synthetic fund's swap and hedged share classes'
        forwards, from the eve of the first trading day."""
        nav = self.portfolio.nav

        if nav.empty:
            return nav

        for owed, sign in ((self.fees.payable, -1.0), (self._swap_value, 1.0),
                           (self._hedge_value, 1.0), (self.dividends.receivable, 1.0)):
            if not owed.empty:
                nav = nav + sign * owed.reindex(nav.index).ffill().fillna(0.0)

        return nav

    def units_on(self,
                 dates: pd.Index) -> pd.Series:
        """Units outstanding on *dates*, carried forward."""
        if self.units.empty:
            return pd.Series(1.0, index=dates)

        return self.units.reindex(dates).ffill().fillna(self.units.iloc[0])

    @property
    def _swap_value(self) -> pd.Series:
        return self.swap.value if self.swap is not None else pd.Series(dtype=float)

    @property
    def _hedge_value(self) -> pd.Series:
        return self.hedging.value if self.hedging is not None else pd.Series(dtype=float)


def fund_record(portfolio: Portfolio,
                currency: str,
                launch_price: float,
                units: pd.Series,
                flows: list[FlowRecord],
                fees_payable: pd.Series,
                dividends_receivable: pd.Series,
                dividends_entitled: pd.DataFrame,
                share_classes: dict[str, ShareClassBook],
                market: pd.DataFrame,
                swap_value: pd.Series,
                swap_resets: list[SwapReset],
                futures_notional: pd.Series,
                futures_trades: list[FuturesTrade],
                futures_rolls: list[FuturesRoll],
                futures_days: pd.DataFrame,
                cash_band_events: pd.DataFrame | None,
                hedge_value: pd.Series) -> FundRecord:
    """A fund's record from what the engine kept, each vehicle part None
    when the run left it empty, and the cash band None when the fund had
    none (*cash_band_events* None)."""
    paid = -sum(flow.amount for flow in portfolio.cash_flows if flow.kind == FEE)
    found = FundRecord(portfolio, currency, launch_price, units, flows,
                       FeesRecord(fees_payable, float(paid)),
                       DividendsRecord(dividends_receivable, dividends_entitled), share_classes)

    def nav() -> pd.Series:
        return found.trading_nav

    if not market.empty:
        found.etf = ETFRecord(market)

    if not swap_value.empty or swap_resets:
        found.swap = SwapRecord(swap_value, swap_resets, nav)

    if not futures_notional.empty or futures_trades or futures_rolls or not futures_days.empty:
        found.futures = FuturesRecord(futures_notional, futures_trades, futures_rolls,
                                      futures_days, nav)

    if cash_band_events is not None:
        found.cash_band = CashBandRecord(cash_band_events)

    if not hedge_value.empty:
        found.hedging = HedgingRecord(hedge_value)

    return found


__all__ = ["CashBandRecord", "DividendsRecord", "ETFRecord", "FeesRecord", "FundRecord",
           "FuturesRecord", "HedgingRecord", "SwapRecord", "fund_record"]
