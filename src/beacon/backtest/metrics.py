# src/beacon/backtest/metrics.py
"""
What a backtest's result says about its performance: returns, risk, tracking
against its index, and, when money flowed, a unit's performance and the
money-weighted return. Mixed into `BacktestResult`.
"""
# Moved out of result.py when it was split (BN-268). The metrics are
# unchanged; they read the result's fields through the declarations below.
import numpy as np
import pandas as pd

from ..assumptions import DEFAULT_PERIODS_PER_YEAR, ModellingAssumptions
from ..portfolio.base import Portfolio
from ..portfolio.cash_flows import DISTRIBUTION, REDEMPTION, SUBSCRIPTION
from .books import IndexBooks
from .flows import FlowRecord, money_weighted_return


class MetricsMixin:
    """Returns, risk and tracking for a backtest's result."""

    # The result's fields, which BacktestResult declares.
    portfolio: Portfolio
    index: IndexBooks
    modelling_assumptions: ModellingAssumptions | None
    flows: list[FlowRecord]
    units: pd.Series
    fees_payable: pd.Series
    launch_price: float

    @property
    def trading_nav(self) -> pd.Series:
        """Provided by BacktestResult; declared so this one type-checks."""
        raise NotImplementedError

    @property
    def aum(self) -> pd.Series:
        """The fund's assets under management each simulated day: the
        trading NAV, which flows move as well as the market."""
        return self.trading_nav

    @property
    def units_outstanding(self) -> pd.Series:
        """Units outstanding at the end of each simulated day."""
        return self._units_on(self.trading_nav.index)

    @property
    def nav_per_unit(self) -> pd.Series:
        """NAV per unit each simulated day: the performance of a unit held
        throughout, whatever money arrived or left. A day with no units
        outstanding carries the last NAV per unit."""
        nav = self.trading_nav
        units = self._units_on(nav.index)
        per_unit = (nav / units).where(units > 0.0)

        return per_unit.ffill().fillna(self.launch_price)

    def _net_nav(self) -> pd.Series:
        """The portfolio's NAV less the fee owed each day."""
        nav = self.portfolio.nav

        if self.fees_payable.empty or nav.empty:
            return nav

        owed = self.fees_payable.reindex(nav.index).ffill().fillna(0.0)

        return nav - owed

    def _units_on(self,
                  dates: pd.Index) -> pd.Series:
        """Units outstanding on *dates*, carried forward."""
        if self.units.empty:
            return pd.Series(1.0, index=dates)

        return self.units.reindex(dates).ffill().fillna(self.units.iloc[0])

    def get_returns(self) -> pd.Series:
        """The portfolio's daily returns.

        The first runs from the initial capital to the first day's close, so
        it carries the cost of the opening trades; each later one is from one
        close to the next. Cash paid out of the book (a ``"distribute"``
        dividend policy) is added back on the day it was paid, so returns
        stay total returns.

        With flows, returns are those of a unit (time-weighted): NAV per unit
        from one day to the next, so money arriving is not a return.

        Returns:
            pd.Series: Returns by date, one per simulated day.
        """
        nav = self.trading_nav

        if nav.empty:
            return pd.Series(dtype=float)

        if self.flows:
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
        nav = self.trading_nav
        units = self._units_on(nav.index)
        paid = self._paid_out().reindex(nav.index, fill_value=0.0)
        per_unit = self.nav_per_unit
        total = ((nav + paid) / units).where(units > 0.0).fillna(per_unit)
        before = per_unit.shift(1)
        before.iloc[0] = self.launch_price

        return (total / before - 1.0).astype(float)

    def performance_levels(self) -> pd.Series:
        """The run's performance as a level series: the trading NAV, or with
        flows the NAV per unit, which they do not move."""
        return self.nav_per_unit if self.flows else self.trading_nav

    def money_weighted_return(self) -> float | None:
        """The annual internal rate of return of the investors' money.

        Counts the initial capital, every subscription and redemption, any
        distribution paid out, and the final NAV, each on its date (ACT/365).
        Unlike the time-weighted figures it depends on when money arrived.

        Returns:
            float or None: The rate a year, or None when it does not exist
            (no money at risk, say).
        """
        nav = self.trading_nav

        if nav.empty:
            return None

        start = (self.portfolio.inception if self.portfolio.inception is not None
                 else nav.index[0])
        investor: list[tuple[pd.Timestamp, float]] = [
            (pd.Timestamp(start), -self.portfolio.initial_capital)]
        investor += [(flow.date, -flow.amount)
                     for flow in self.portfolio.cash_flows
                     if flow.kind in (SUBSCRIPTION, REDEMPTION, DISTRIBUTION)]
        investor.append((pd.Timestamp(nav.index[-1]), float(nav.iloc[-1])))

        return money_weighted_return([(date, amount) for date, amount in investor
                                      if amount != 0.0])

    def _paid_out(self) -> pd.Series:
        """Cash paid out of the book to its investors, by date, as positive
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
        nav = self.trading_nav

        if self.flows and not nav.empty:
            launch = self.launch_price

            return launch * (1.0 + self.get_returns()).cumprod(), launch

        initial = self.portfolio.initial_capital

        if self._paid_out().empty or nav.empty:
            return nav, initial

        return initial * (1.0 + self.get_returns()).cumprod(), initial

    def get_annual_returns(self) -> pd.Series:
        """The portfolio's calendar-year returns.

        Each year runs from the previous year's last close, and the first from
        the initial capital, so the years compound to the whole run's return.
        A partial first or last year covers only its days.

        Returns:
            pd.Series: Return by calendar year, indexed by the year as an int.
        """
        returns = self.get_returns()

        if returns.empty:
            return pd.Series(dtype=float)

        yearly = (1.0 + returns).groupby(returns.index.year).prod() - 1.0

        return yearly.astype(float)

    def _periods_per_year(self) -> int:
        """The run's annualisation factor."""
        assumptions = self.modelling_assumptions
        periods = assumptions.periods_per_year if assumptions else None

        return periods or DEFAULT_PERIODS_PER_YEAR

    def _risk_free_rate(self) -> float:
        """The rate the run's Sharpe ratio is measured against."""
        assumptions = self.modelling_assumptions
        rate = assumptions.risk_free_rate if assumptions else None

        return rate or 0.0

    def get_tracking_error(self) -> float | None:
        """Calculate annualised tracking error against the tracked index.

        Tracking error is the standard deviation of the difference between
        daily portfolio returns and index returns on their common dates,
        annualised by the square root of the run's periods per year (252
        unless its modelling assumptions say otherwise). The first day
        compares the
        portfolio's return from its initial capital (the opening trades'
        cost) with the index's return of zero from its starting level.

        Returns:
            float or None: Annualised tracking error, or None if the run
            tracked no index.
        """
        tracked = self.index.tracked
        if tracked is None:
            return None

        port_returns = self.get_returns()
        index_returns = _from_start(tracked.levels)

        # Align on common dates
        aligned = pd.DataFrame({
            "port": port_returns,
            "index": index_returns,
        }).dropna()

        if aligned.empty:
            return None

        active_returns = aligned["port"] - aligned["index"]
        return float(active_returns.std() * np.sqrt(self._periods_per_year()))

    def get_tracking_difference(self) -> float | None:
        """Calculate cumulative tracking difference against the tracked index.

        Tracking difference is the portfolio's cumulative return, from its
        initial capital, less the index's cumulative return, from its level on
        the first day, over the whole run. The cost of the opening trades is
        in it.

        Returns:
            float or None: Tracking difference, or None if the run tracked
            no index.
        """
        tracked = self.index.tracked
        if tracked is None:
            return None

        port_returns = self.get_returns()
        index_returns = _from_start(tracked.levels)

        if port_returns.empty or index_returns.empty:
            return None

        port_cumulative = (1 + port_returns).prod() - 1
        index_cumulative = (1 + index_returns).prod() - 1
        return float(port_cumulative - index_cumulative)

    def summary(self) -> dict[str, float | None]:
        """Calculate key performance metrics for the backtest.

        Returns are daily and annualised over the run's periods per year, and
        the Sharpe ratio is the annualised return in excess of its risk-free
        rate, over the volatility. Both come from the run's modelling
        assumptions: 252 and 0 unless they say otherwise. Total return is
        measured against the portfolio's initial capital, with any dividends
        paid out of the book added back.

        With flows every figure is a unit's, and the money-weighted return
        is added.

        Returns:
            dict: Dictionary containing: total_return, annualised_return,
            volatility, sharpe_ratio, max_drawdown, and, when the run tracked
            an index, tracking_error and tracking_difference; with flows,
            money_weighted_return.
        """
        returns = self.get_returns()
        n_periods = len(returns)
        # BN-263: with dividends paid out, the NAV understates performance;
        # BN-267: with flows it is the fund's size, so a unit is measured.
        nav, initial = self._performance_nav()

        # Total return
        total_return = (0.0 if nav.empty or initial == 0
                        else float(nav.iloc[-1] / initial - 1))

        # Annualised return
        periods = self._periods_per_year()

        if n_periods > 0:
            years = n_periods / float(periods)
            annualised_return = float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0
        else:
            annualised_return = 0.0

        # Volatility (annualised)
        volatility = float(returns.std() * np.sqrt(periods)) if n_periods > 1 else 0.0

        # BN-276: in excess of the run's risk-free rate, which was always 0.
        risk_free = self._risk_free_rate()
        sharpe_ratio = (float((annualised_return - risk_free) / volatility)
                        if volatility > 0 else 0.0)

        # Max drawdown, from the initial capital, so a fall on the first day
        # (the cost of buying in, say) counts.
        if not nav.empty:
            if initial > 0:
                nav = pd.concat([pd.Series([initial]), nav], ignore_index=True)

            cumulative_max = nav.cummax()
            drawdown = (nav - cumulative_max) / cumulative_max
            max_drawdown = float(drawdown.min())
        else:
            max_drawdown = 0.0

        result: dict[str, float | None] = {
            "total_return": total_return,
            "annualised_return": annualised_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }

        # Tracking metrics (only if the run tracked an index)
        te = self.get_tracking_error()
        td = self.get_tracking_difference()
        if te is not None:
            result["tracking_error"] = te
        if td is not None:
            result["tracking_difference"] = td

        if self.flows:
            result["money_weighted_return"] = self.money_weighted_return()

        return result


def _from_start(levels: pd.Series) -> pd.Series:
    """Daily returns of a level series, the first day's being zero.

    The index starts at its level on the first day, so its return that day is
    nothing; the portfolio's first return, from its capital, lines up with it.
    """
    returns = levels.pct_change()

    if not returns.empty:
        returns.iloc[0] = 0.0

    return returns.dropna()
