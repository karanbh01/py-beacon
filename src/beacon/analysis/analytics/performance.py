# src/beacon/analysis/analytics/performance.py
"""
A run's performance analytics: returns, risk-adjusted ratios, rolling
outcomes, and crises and regimes.

    performance = result.analytics.portfolio.performance
    performance.total_return()                   # the fund's, over the run
    performance.sharpe_ratio(of="all", by="year")
    performance.crisis_returns(against="benchmark")
"""
# BN-435 (Karan, 2026-11-24): each method is its catalogue statistic, worked
# out as the run's analysis works it out.
from typing import Any

from ..reports.context import FUND
from .category import Category, Day, Windows


class PerformanceCategory(Category):
    """A run's performance analytics: returns, risk-adjusted ratios, rolling
    outcomes, and crises and regimes. Each method is a statistic, named by
    its id; its keywords pick the form it is given in
    (`beacon.analysis.analytics.forms`)."""

    name = "performance"
    statistics = ("total_return", "annualised_return", "money_weighted_return", "mean_return",
                  "best_and_worst", "positive_share", "hit_rate", "gains_and_losses", "streaks",
                  "month_of_year", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
                  "omega_ratio", "kappa_3", "upside_potential_ratio", "martin_ratio",
                  "sterling_ratio", "burke_ratio", "gain_to_pain", "probabilistic_sharpe_ratio",
                  "deflated_sharpe_ratio", "lo_sharpe_ratio", "rolling_outcomes",
                  "crisis_returns", "crisis_paths", "crisis_summary", "regimes")
    extras = ("returns",)

    # -- returns --------------------------------------------------------------

    def total_return(self,
                     *,
                     of: str = FUND,
                     start: Day = None,
                     end: Day = None,
                     by: str | None = None,
                     trailing: Windows = None,
                     rolling: int | str | None = None,
                     step: str | None = None) -> Any:
        """The fund's compounded return over the window."""
        return self._get("total_return", of=of, start=start, end=end, by=by, trailing=trailing,
                         rolling=rolling, step=step)

    def annualised_return(self,
                          *,
                          of: str = FUND,
                          start: Day = None,
                          end: Day = None,
                          by: str | None = None,
                          trailing: Windows = None,
                          rolling: int | str | None = None,
                          step: str | None = None,
                          annualise: bool | None = None) -> Any:
        """The constant yearly rate that compounds to the total return."""
        return self._get("annualised_return", of=of, start=start, end=end, by=by,
                         trailing=trailing, rolling=rolling, step=step, annualise=annualise)

    def money_weighted_return(self,
                              *,
                              start: Day = None,
                              end: Day = None) -> Any:
        """The return an investor earned given when money came in and went
        out (the IRR)."""
        return self._get("money_weighted_return", start=start, end=end)

    def mean_return(self,
                    *,
                    of: str = FUND,
                    start: Day = None,
                    end: Day = None,
                    rolling: int | str | None = None,
                    step: str | None = None) -> Any:
        """The mean daily return, and the daily rate that compounds to the
        total, each also a year."""
        return self._get("mean_return", of=of, start=start, end=end, rolling=rolling, step=step)

    def best_and_worst(self,
                       *,
                       of: str = FUND,
                       start: Day = None,
                       end: Day = None) -> Any:
        """The best and worst day, week, month, quarter and year."""
        return self._get("best_and_worst", of=of, start=start, end=end)

    def positive_share(self,
                       *,
                       of: str = FUND,
                       start: Day = None,
                       end: Day = None,
                       by: str | None = None,
                       rolling: int | str | None = None,
                       step: str | None = None) -> Any:
        """The share of days, months and years with a gain."""
        return self._get("positive_share", of=of, start=start, end=end, by=by, rolling=rolling,
                         step=step)

    def hit_rate(self,
                 *,
                 of: str = FUND,
                 start: Day = None,
                 end: Day = None,
                 by: str | None = None,
                 rolling: int | str | None = None,
                 step: str | None = None) -> Any:
        """The share of months the fund rose."""
        return self._get("hit_rate", of=of, start=start, end=end, by=by, rolling=rolling, step=step)

    def gains_and_losses(self,
                         *,
                         of: str = FUND,
                         start: Day = None,
                         end: Day = None,
                         rolling: int | str | None = None,
                         step: str | None = None) -> Any:
        """The mean of the periods that gained and of those that lost, and
        their ratio."""
        return self._get("gains_and_losses", of=of, start=start, end=end, rolling=rolling,
                         step=step)

    def streaks(self,
                *,
                of: str = FUND,
                start: Day = None,
                end: Day = None) -> Any:
        """The most months in a row with a gain, and with a loss."""
        return self._get("streaks", of=of, start=start, end=end)

    def month_of_year(self) -> Any:
        """The mean return of each calendar month across the years."""
        return self._get("month_of_year")

    def returns(self,
                *,
                of: str = FUND,
                start: Day = None,
                end: Day = None) -> Any:
        """Daily returns over the window: the fund's, a portfolio's, or each
        portfolio's (``of="all"``), a column each."""
        return self._analytics._returns(of, start, end)

    # -- risk-adjusted --------------------------------------------------------

    def sharpe_ratio(self,
                     *,
                     of: str = FUND,
                     start: Day = None,
                     end: Day = None,
                     by: str | None = None,
                     trailing: Windows = None,
                     rolling: int | str | None = None,
                     step: str | None = None,
                     annualise: bool | None = None) -> Any:
        """Mean daily return over the risk-free rate, per unit of its
        volatility, annualised."""
        return self._get("sharpe_ratio", of=of, start=start, end=end, by=by, trailing=trailing,
                         rolling=rolling, step=step, annualise=annualise)

    def sortino_ratio(self,
                      *,
                      of: str = FUND,
                      start: Day = None,
                      end: Day = None,
                      rolling: int | str | None = None,
                      step: str | None = None,
                      annualise: bool | None = None) -> Any:
        """The Sharpe ratio with only downside deviation as risk."""
        return self._get("sortino_ratio", of=of, start=start, end=end, rolling=rolling, step=step,
                         annualise=annualise)

    def calmar_ratio(self,
                     *,
                     of: str = FUND,
                     start: Day = None,
                     end: Day = None,
                     rolling: int | str | None = None,
                     step: str | None = None,
                     annualise: bool | None = None) -> Any:
        """The annualised return over the deepest drawdown."""
        return self._get("calmar_ratio", of=of, start=start, end=end, rolling=rolling, step=step,
                         annualise=annualise)

    def omega_ratio(self,
                    *,
                    of: str = FUND,
                    start: Day = None,
                    end: Day = None,
                    rolling: int | str | None = None,
                    step: str | None = None) -> Any:
        """Gains above the daily risk-free rate over losses below it."""
        return self._get("omega_ratio", of=of, start=start, end=end, rolling=rolling, step=step)

    def kappa_3(self,
                *,
                of: str = FUND,
                start: Day = None,
                end: Day = None,
                rolling: int | str | None = None,
                step: str | None = None) -> Any:
        """Excess return over the third lower partial moment."""
        return self._get("kappa_3", of=of, start=start, end=end, rolling=rolling, step=step)

    def upside_potential_ratio(self,
                               *,
                               of: str = FUND,
                               start: Day = None,
                               end: Day = None,
                               rolling: int | str | None = None,
                               step: str | None = None) -> Any:
        """Mean gain above the target over downside deviation."""
        return self._get("upside_potential_ratio", of=of, start=start, end=end, rolling=rolling,
                         step=step)

    def martin_ratio(self,
                     *,
                     of: str = FUND,
                     start: Day = None,
                     end: Day = None,
                     rolling: int | str | None = None,
                     step: str | None = None,
                     annualise: bool | None = None) -> Any:
        """Excess annual return over the Ulcer index."""
        return self._get("martin_ratio", of=of, start=start, end=end, rolling=rolling, step=step,
                         annualise=annualise)

    def sterling_ratio(self,
                       *,
                       of: str = FUND,
                       start: Day = None,
                       end: Day = None,
                       rolling: int | str | None = None,
                       step: str | None = None,
                       annualise: bool | None = None) -> Any:
        """Annual return over the average of the yearly maximum
        drawdowns."""
        return self._get("sterling_ratio", of=of, start=start, end=end, rolling=rolling,
                         step=step, annualise=annualise)

    def burke_ratio(self,
                    *,
                    of: str = FUND,
                    start: Day = None,
                    end: Day = None,
                    rolling: int | str | None = None,
                    step: str | None = None,
                    annualise: bool | None = None) -> Any:
        """Excess annual return over the root sum of squared drawdowns."""
        return self._get("burke_ratio", of=of, start=start, end=end, rolling=rolling, step=step,
                         annualise=annualise)

    def gain_to_pain(self,
                     *,
                     of: str = FUND,
                     start: Day = None,
                     end: Day = None,
                     rolling: int | str | None = None,
                     step: str | None = None) -> Any:
        """The sum of monthly returns over the sum of monthly losses."""
        return self._get("gain_to_pain", of=of, start=start, end=end, rolling=rolling, step=step)

    def probabilistic_sharpe_ratio(self,
                                   *,
                                   of: str = FUND,
                                   start: Day = None,
                                   end: Day = None,
                                   rolling: int | str | None = None,
                                   step: str | None = None) -> Any:
        """The probability the true Sharpe ratio is above the one to beat,
        given the sample's length, skew and kurtosis."""
        return self._get("probabilistic_sharpe_ratio", of=of, start=start, end=end, rolling=rolling,
                         step=step)

    def deflated_sharpe_ratio(self,
                              *,
                              of: str = FUND,
                              start: Day = None,
                              end: Day = None,
                              rolling: int | str | None = None,
                              step: str | None = None) -> Any:
        """The probabilistic Sharpe ratio against the best of many trials by
        chance."""
        return self._get("deflated_sharpe_ratio", of=of, start=start, end=end, rolling=rolling,
                         step=step)

    def lo_sharpe_ratio(self,
                        *,
                        of: str = FUND,
                        start: Day = None,
                        end: Day = None,
                        rolling: int | str | None = None,
                        step: str | None = None,
                        annualise: bool | None = None) -> Any:
        """The Sharpe ratio corrected for serial correlation when
        annualising."""
        return self._get("lo_sharpe_ratio", of=of, start=start, end=end, rolling=rolling,
                         step=step, annualise=annualise)

    # -- rolling --------------------------------------------------------------

    def rolling_outcomes(self,
                         *,
                         start: Day = None,
                         end: Day = None,
                         against: str | None = None) -> Any:
        """The worst, median and best rolling return, the share of windows
        that gained, and the share ahead of the tracked comparator."""
        return self._get("rolling_outcomes", start=start, end=end, against=against)

    # -- crises and regimes ---------------------------------------------------

    def crisis_returns(self,
                       *,
                       against: str | None = None) -> Any:
        """The fund's and comparator's return, outperformance, drawdowns,
        beta and alpha in each crisis."""
        return self._get("crisis_returns", against=against)

    def crisis_paths(self) -> Any:
        """The fund's, the index's and the benchmark's levels through each
        crisis and the months after it, rebased to 100 at its start."""
        return self._get("crisis_paths")

    def crisis_summary(self,
                       *,
                       start: Day = None,
                       end: Day = None,
                       against: str | None = None) -> Any:
        """Across crises: average outperformance, share outperformed,
        capture, pooled alpha."""
        return self._get("crisis_summary", start=start, end=end, against=against)

    def regimes(self,
                *,
                start: Day = None,
                end: Day = None,
                against: str | None = None) -> Any:
        """Returns and relative figures in up and down months, calm and
        volatile months, bear markets and corrections."""
        return self._get("regimes", start=start, end=end, against=against)
