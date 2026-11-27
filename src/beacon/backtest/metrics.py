# src/beacon/backtest/metrics.py
"""
What a backtest's result says about its performance: returns, risk, tracking
against its index, and, when money flowed, a unit's performance and the
money-weighted return. Mixed into `BacktestResult`.
"""
# Moved out of result.py when it was split (BN-268). The metrics are
# unchanged; they read the result's fields through the declarations below.
import warnings
from typing import Any

import numpy as np
import pandas as pd

from ..analysis.attribution import AttributionResult, attribute
from ..analysis.performance import formulas
from ..assumptions import DEFAULT_PERIODS_PER_YEAR, ModellingAssumptions
from ..exceptions import CalculationError
from ..portfolio.base import Portfolio
from ..portfolio.cash_flows import DISTRIBUTION, REDEMPTION, SUBSCRIPTION
from .books import IndexRecords
from .fund_record import FundRecord
from .trading_record import TradingRecord


class MetricsMixin:
    """Returns, risk and tracking for a backtest's result."""

    # The result's data, which BacktestResult holds (BN-459/BN-460).
    portfolio: Portfolio
    index: IndexRecords
    assumptions: ModellingAssumptions | None
    fund: FundRecord
    trading: TradingRecord

    def _market(self) -> pd.DataFrame:
        """An ETF's quotes; empty for another vehicle."""
        return self.fund.etf.market if self.fund.etf is not None else pd.DataFrame()

    def _returns(self) -> pd.Series:
        """The portfolio's daily returns.

        The first runs from the initial capital to the first day's close, so
        it carries the cost of the opening trades; each later one is from one
        close to the next. Cash paid out of the portfolio (a ``"distribute"``
        dividend policy) is added back on the day it was paid, so returns
        stay total returns.

        With flows, returns are those of a unit (time-weighted): NAV per unit
        from one day to the next, so money arriving is not a return.

        Returns:
            pd.Series: Returns by date, one per simulated day.
        """
        nav = self.fund.trading_nav

        if nav.empty:
            return pd.Series(dtype=float)

        if self.fund.flows:
            return self._unit_returns()

        paid = self._paid_out().reindex(nav.index, fill_value=0.0)
        returns = (nav + paid) / nav.shift(1) - 1.0
        initial = self.portfolio.initial_capital

        # BN-246: the first return used to be dropped, so the cost of buying
        # in on day one never reached any metric.
        if initial > 0:
            returns.iloc[0] = (nav.iloc[0] + paid.iloc[0]) / initial - 1.0

            return returns

        return returns.dropna()

    def _unit_returns(self) -> pd.Series:
        """Daily returns of NAV per unit, with cash paid out added back."""
        nav = self.fund.trading_nav
        units = self.fund.units_on(nav.index)
        paid = self._paid_out().reindex(nav.index, fill_value=0.0)
        per_unit = self.fund.nav_per_unit
        total = ((nav + paid) / units).where(units > 0.0).fillna(per_unit)
        before = per_unit.shift(1)
        before.iloc[0] = self.fund.launch_price

        return (total / before - 1.0).astype(float)

    def _performance_levels(self) -> pd.Series:
        """The run's performance as a level series: the trading NAV, or with
        flows the NAV per unit, which they do not move."""
        return self.fund.nav_per_unit if self.fund.flows else self.fund.trading_nav

    def _money_weighted_return(self) -> float | None:
        """The annual internal rate of return of the investors' money.

        Counts the initial capital, every subscription and redemption, any
        distribution paid out, and the final NAV, each on its date (ACT/365).
        Unlike the time-weighted figures it depends on when money arrived.

        Returns:
            float or None: The rate a year, or None when it does not exist
            (no money at risk, say).
        """
        flows = self._investor_flows()

        if flows.empty:
            return None

        return formulas.money_weighted_return([(pd.Timestamp(date), float(amount))
                                               for date, amount in flows.items()
                                               if amount != 0.0])

    def _investor_flows(self) -> pd.Series:
        """The investors' cash flows, dated, from their side: the initial
        capital, every subscription, redemption and distribution, and the
        final NAV. Empty with no NAV."""
        nav = self.fund.trading_nav

        if nav.empty:
            return pd.Series(dtype=float)

        start = (self.portfolio.inception if self.portfolio.inception is not None
                 else nav.index[0])
        investor: list[tuple[pd.Timestamp, float]] = [
            (pd.Timestamp(start), -self.portfolio.initial_capital)]
        investor += [(flow.date, -flow.amount)
                     for flow in self.portfolio.cash_flows
                     if flow.kind in (SUBSCRIPTION, REDEMPTION, DISTRIBUTION)]
        investor.append((pd.Timestamp(nav.index[-1]), float(nav.iloc[-1])))

        return pd.Series([amount for _, amount in investor],
                         index=pd.DatetimeIndex([date for date, _ in investor]), dtype=float)

    def _paid_out(self) -> pd.Series:
        """Cash paid out of the portfolio to its investors, by date, as positive
        amounts. Empty unless the run distributed dividends."""
        paid = [(flow.date, -flow.amount) for flow in self.portfolio.cash_flows
                if flow.kind == DISTRIBUTION]

        if not paid:
            return pd.Series(dtype=float)

        frame = pd.DataFrame(paid, columns=["date", "amount"])

        return frame.groupby("date")["amount"].sum()

    def _performance_nav(self) -> tuple[pd.Series, float]:
        """The NAV as total return, and what it started from: the NAV and the
        initial capital; the capital grown by the returns, when cash was
        paid out; or, with flows, a unit grown from its launch price."""
        nav = self.fund.trading_nav

        if self.fund.flows and not nav.empty:
            launch = self.fund.launch_price

            return launch * (1.0 + self._returns()).cumprod(), launch

        initial = self.portfolio.initial_capital

        if self._paid_out().empty or nav.empty:
            return nav, initial

        return initial * (1.0 + self._returns()).cumprod(), initial

    def _annual_returns(self) -> pd.Series:
        """The portfolio's calendar-year returns.

        Each year runs from the previous year's last close, and the first from
        the initial capital, so the years compound to the whole run's return.
        A partial first or last year covers only its days.

        Returns:
            pd.Series: Return by calendar year, indexed by the year as an int.
        """
        returns = self._returns()

        if returns.empty:
            return pd.Series(dtype=float)

        yearly = formulas.period_returns(returns, formulas.ANNUAL)
        yearly.index = yearly.index.astype(int)

        return yearly.astype(float)

    def _periods_per_year(self) -> int:
        """The run's annualisation factor."""
        assumptions = self.assumptions
        periods = assumptions.periods_per_year if assumptions else None

        return periods or DEFAULT_PERIODS_PER_YEAR

    def _risk_free_rate(self) -> float:
        """The rate the run's Sharpe ratio is measured against."""
        assumptions = self.assumptions
        rate = assumptions.risk_free_rate if assumptions else None

        return rate or 0.0

    def _annualising(self) -> tuple[str, str]:
        """The shortest windows the run's annualised rates and ratios are
        given over (`beacon.analysis.annualising`)."""
        assumptions = self.assumptions or ModellingAssumptions()
        resolved = assumptions.with_defaults()

        return (str(resolved.min_annualised_rate_window),
                str(resolved.min_annualised_ratio_window))

    def _analytics(self) -> Any:
        """The run's performance analytics, which its figures are."""
        # Imported here, as both read the result this mixin is part of, which
        # is why it passes itself as one (BN-371).
        from ..analysis.data import RunData  # noqa: PLC0415
        from ..analysis.performance.analytics import PerformanceAnalytics  # noqa: PLC0415

        return PerformanceAnalytics(RunData.from_result(self))  # type: ignore[arg-type]

    def _tracking_error(self) -> float | None:
        """Calculate annualised tracking error against the tracked index.

        Tracking error is the standard deviation of the difference between
        daily portfolio returns and index returns on their common dates,
        annualised by the square root of the run's periods per year (252
        unless its modelling assumptions say otherwise). The first day
        compares the portfolio's return from its initial capital (the opening
        trades' cost) with the index's return of zero from its starting
        level.

        Returns:
            float or None: Annualised tracking error, or None if the run
            tracked no index or they share fewer than two days.
        """
        return self._analytics().tracking_error()  # type: ignore[no-any-return]

    def _tracking_difference(self) -> float | None:
        """Calculate cumulative tracking difference against the tracked index.

        Tracking difference is the portfolio's cumulative return, from its
        initial capital, less the index's cumulative return, from its level on
        the first day, over the whole run. The cost of the opening trades is
        in it.

        Returns:
            float or None: Tracking difference, or None if the run tracked
            no index.
        """
        return self._analytics().tracking_difference()  # type: ignore[no-any-return]

    def summary(self) -> dict[str, float | None]:
        """Calculate key performance metrics for the backtest.

        Returns are daily and annualised over the run's periods per year, and
        the Sharpe ratio is the annualised return in excess of its risk-free
        rate, over the volatility. Both come from the run's modelling
        assumptions: 252 and 0 unless they say otherwise. Total return is
        measured against the portfolio's initial capital, with any dividends
        paid out of the portfolio added back.

        With flows every figure is a unit's, and the money-weighted return
        is added. The figures are `beacon.analysis.performance.PerformanceAnalytics`'s
        over the whole run, with an ETF's trading and an active strategy's
        own figures added.

        Returns:
            dict: Dictionary containing: total_return, annualised_return,
            volatility, sharpe_ratio, max_drawdown, hit_rate_monthly, and,
            when the run tracked an index, tracking_error and
            tracking_difference; with flows, money_weighted_return.
        """
        # BN-371: through the run's analysis, which a stored run is measured
        # with too.
        result: dict[str, float | None] = self._analytics().summary()

        if not self._market().empty:
            result.update(self._market_summary())

        if self.trading.active:
            result.update(self._active_figures())

        return result

    def _information_ratio(self) -> float | None:
        """Annualised active return over annualised tracking error, against
        the tracked index. None without an index, or with no active risk."""
        return self._analytics().information_ratio()  # type: ignore[no-any-return]

    def _active_share(self) -> pd.Series:
        """An active strategy's active share at each rebalance."""
        return pd.Series({step.date: step.active_share for step in (self.trading.active or [])},
                         dtype=float)

    def active_summary(self) -> dict[str, float | None]:
        """An active strategy's figures: its information ratio, average
        active share and average ex-ante tracking error.

        Deprecated: each is a figure of `result.analytics.portfolio.relative`:
        ``information_ratio()``, ``active_share().mean()`` and
        ``ex_ante_tracking_error().mean()``.
        """
        # BN-451: removed in 0.12.0.
        warnings.warn("active_summary() is deprecated and will be removed in 0.12.0; use "
                      "result.analytics.portfolio.relative.information_ratio(), "
                      "active_share().mean() and ex_ante_tracking_error().mean().",
                      DeprecationWarning, stacklevel=2)

        return self._active_figures()

    def _active_figures(self) -> dict[str, float | None]:
        """An active strategy's information ratio, average active share and
        average ex-ante tracking error, as its summary gives them."""
        return {"information_ratio": self._information_ratio(),
                "average_active_share": float(np.mean([step.active_share
                                                       for step in (self.trading.active or [])])),
                "average_ex_ante_tracking_error": float(np.mean(
                    [step.tracking_error for step in (self.trading.active or [])]))}

    def _active_attribution(self) -> AttributionResult:
        """Which names produced the active return: each name's active weight
        (the portfolio's less the index's, as both drift) times its return,
        linked over the run so they add up to the portfolio's compounded
        return less the index's. Costs, cash and fees are in the residual.

        Raises:
            CalculationError: If the run has no index to compare with, or
                the weights and returns share no dates.
        """
        tracked = self.index.tracked

        if tracked is None or tracked.weights.empty:
            raise CalculationError("Attribution", "the run tracked no index, so "
                                   "there is no active return to attribute.")

        held = self.portfolio.weights.fillna(0.0)
        index = tracked.weights.reindex(held.index).ffill().fillna(0.0)
        names = sorted(set(held.columns) | set(index.columns))
        active_weights = (held.reindex(columns=names, fill_value=0.0)
                          - index.reindex(columns=names, fill_value=0.0))
        prices = self.portfolio.source.fetch_prices(
            names, str(held.index[0].date()), str(held.index[-1].date()),
            currency=self.fund.currency) if self.portfolio.source is not None else None

        if prices is None or prices.empty:
            raise CalculationError("Attribution", "no prices to measure the "
                                   "names' returns with.")

        ours = self._returns().dropna()
        theirs = formulas.returns_from_start(tracked.levels).dropna()
        days = ours.index.intersection(theirs.index)

        # Linked by the active Carino factor, so the contributions add up to
        # the portfolio's compounded return less the index's.
        return attribute(ours.loc[days], active_weights,
                         prices.ffill().pct_change(fill_method=None),
                         comparator=theirs.loc[days])

    def _market_return(self) -> float | None:
        """An exchange investor's return on an ETF: bought at the ask on the
        first day and sold at the bid on the last. None for another vehicle.
        Cash the fund paid out is not included."""
        if self._market().empty:
            return None

        return formulas.market_return(float(self._market()["ask"].iloc[0]),
                                      float(self._market()["bid"].iloc[-1]))

    def _market_summary(self) -> dict[str, float | None]:
        """An ETF's trading on the exchange: the investor's return at market
        prices, the average premium and spread, and the share of days at a
        premium."""
        market = self._market()

        return {"market_return": self._market_return(),
                "average_premium": float(market["premium"].mean()),
                "average_spread": float(market["spread"].mean()),
                "days_at_premium": float((market["premium"] > 0).mean())}
