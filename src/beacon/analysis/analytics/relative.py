# src/beacon/analysis/analytics/relative.py
"""
A run's relative analytics: the fund against its index or benchmark, which
against= picks: tracking, regression, capture and active share.

    relative = result.analytics.portfolio.relative
    relative.tracking_error()                    # against the tracked index
    relative.beta(against="benchmark", by="year")
"""
# BN-437 (Karan, 2026-11-24): each method is its catalogue statistic, worked
# out as the run's analysis works it out.
from typing import Any

import pandas as pd

from .category import Category, Day, Windows


class RelativeCategory(Category):
    """A run's relative analytics: the fund against its index or benchmark,
    which against= picks: tracking, regression, capture and active share.
    Each method is a statistic, named by its id; its keywords pick the form
    it is given in (`beacon.analysis.analytics.forms`)."""

    name = "relative"
    statistics = ("tracking_difference", "tracking_error", "information_ratio",
                  "information_ratio_t", "annualised_relative_return", "beta",
                  "correlation", "alpha", "r_squared", "residual_risk",
                  "treynor_ratio", "capture", "batting_average",
                  "downside_tracking_error", "relative_drawdowns")
    extras = ("active_share",)

    # -- against the index or benchmark ---------------------------------------

    def tracking_difference(self,
                            *,
                            start: Day = None,
                            end: Day = None,
                            by: str | None = None,
                            trailing: Windows = None,
                            rolling: int | str | None = None,
                            step: str | None = None,
                            against: str | None = None) -> Any:
        """The fund's return less the comparator's over the window."""
        return self._get("tracking_difference", start=start, end=end, by=by, trailing=trailing,
                         rolling=rolling, step=step, against=against)

    def tracking_error(self,
                       *,
                       start: Day = None,
                       end: Day = None,
                       by: str | None = None,
                       trailing: Windows = None,
                       rolling: int | str | None = None,
                       step: str | None = None,
                       against: str | None = None) -> Any:
        """The annualised standard deviation of daily active returns."""
        return self._get("tracking_error", start=start, end=end, by=by, trailing=trailing,
                         rolling=rolling, step=step, against=against)

    def information_ratio(self,
                          *,
                          start: Day = None,
                          end: Day = None,
                          by: str | None = None,
                          rolling: int | str | None = None,
                          step: str | None = None,
                          against: str | None = None) -> Any:
        """Mean daily active return over its spread, annualised."""
        return self._get("information_ratio", start=start, end=end, by=by, rolling=rolling,
                         step=step, against=against)

    def information_ratio_t(self,
                            *,
                            start: Day = None,
                            end: Day = None,
                            rolling: int | str | None = None,
                            step: str | None = None,
                            against: str | None = None) -> Any:
        """How many standard errors the mean active return is from zero, and
        the chance of that."""
        return self._get("information_ratio_t", start=start, end=end, rolling=rolling, step=step,
                         against=against)

    def annualised_relative_return(self,
                                   *,
                                   start: Day = None,
                                   end: Day = None,
                                   by: str | None = None,
                                   rolling: int | str | None = None,
                                   step: str | None = None,
                                   against: str | None = None) -> Any:
        """The yearly rate of the fund's growth relative to the
        comparator's."""
        return self._get("annualised_relative_return", start=start, end=end, by=by, rolling=rolling,
                         step=step, against=against)

    def beta(self,
             *,
             start: Day = None,
             end: Day = None,
             by: str | None = None,
             rolling: int | str | None = None,
             step: str | None = None,
             against: str | None = None) -> Any:
        """How much the fund moves with the comparator."""
        return self._get("beta", start=start, end=end, by=by, rolling=rolling, step=step,
                         against=against)

    def correlation(self,
                    *,
                    start: Day = None,
                    end: Day = None,
                    by: str | None = None,
                    rolling: int | str | None = None,
                    step: str | None = None,
                    against: str | None = None) -> Any:
        """How closely the fund's daily returns move with the
        comparator's."""
        return self._get("correlation", start=start, end=end, by=by, rolling=rolling, step=step,
                         against=against)

    def alpha(self,
              *,
              start: Day = None,
              end: Day = None,
              rolling: int | str | None = None,
              step: str | None = None,
              against: str | None = None) -> Any:
        """The intercept of the fund's excess returns regressed on the
        comparator's, a year, and its significance."""
        return self._get("alpha", start=start, end=end, rolling=rolling, step=step, against=against)

    def r_squared(self,
                  *,
                  start: Day = None,
                  end: Day = None,
                  rolling: int | str | None = None,
                  step: str | None = None,
                  against: str | None = None) -> Any:
        """The share of the fund's return variance the comparator
        explains."""
        return self._get("r_squared", start=start, end=end, rolling=rolling, step=step,
                         against=against)

    def residual_risk(self,
                      *,
                      start: Day = None,
                      end: Day = None,
                      rolling: int | str | None = None,
                      step: str | None = None,
                      against: str | None = None) -> Any:
        """The annualised spread of the regression's residuals."""
        return self._get("residual_risk", start=start, end=end, rolling=rolling, step=step,
                         against=against)

    def treynor_ratio(self,
                      *,
                      start: Day = None,
                      end: Day = None,
                      rolling: int | str | None = None,
                      step: str | None = None,
                      against: str | None = None) -> Any:
        """Excess annual return per unit of beta."""
        return self._get("treynor_ratio", start=start, end=end, rolling=rolling, step=step,
                         against=against)

    def capture(self,
                *,
                start: Day = None,
                end: Day = None,
                rolling: int | str | None = None,
                step: str | None = None,
                against: str | None = None) -> Any:
        """The fund's mean return in the comparator's up months (and down
        months) over the comparator's."""
        return self._get("capture", start=start, end=end, rolling=rolling, step=step,
                         against=against)

    def batting_average(self,
                        *,
                        start: Day = None,
                        end: Day = None,
                        by: str | None = None,
                        rolling: int | str | None = None,
                        step: str | None = None,
                        against: str | None = None) -> Any:
        """The share of months the fund beat the comparator."""
        return self._get("batting_average", start=start, end=end, by=by, rolling=rolling, step=step,
                         against=against)

    def downside_tracking_error(self,
                                *,
                                start: Day = None,
                                end: Day = None,
                                rolling: int | str | None = None,
                                step: str | None = None,
                                against: str | None = None) -> Any:
        """Tracking error counting only days the fund lagged."""
        return self._get("downside_tracking_error", start=start, end=end, rolling=rolling,
                         step=step, against=against)

    def relative_drawdowns(self,
                           *,
                           against: str | None = None) -> Any:
        """Drawdowns of the fund's value relative to the tracked
        comparator's."""
        return self._get("relative_drawdowns", against=against)

    def active_share(self) -> pd.Series:
        """An active strategy's active share against its index at each
        rebalance: half the summed absolute differences of their weights."""
        return self._analytics._active_share()

