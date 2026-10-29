# src/beacon/analysis/performance/crises.py
"""
How a run held up in crises: each crisis window's returns, outperformance,
drawdowns, beta and alpha, what came after the trough and how long the fund
took to recover; and the figures across every crisis.

    crises = CrisisAnalytics(RunData.from_record(folder), market=sp500_levels)
    crises.crises(), crises.summary()

The windows come from a broad market series by default: each fall of the
market deeper than a threshold (15%), from its peak to its trough. Or they
are the user's own named episodes, each a start and an end date, instead. A
window is measured over the days after its start through its end; one the run
covers only part of is marked partial.
"""
# BN-380. As decided with Karan on 2026-10-27: the threshold applies to a
# broad market series the user chooses, market data or an index built in
# Beacon, not necessarily the fund's comparator.
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import BENCHMARK, INDEX, RunData
from . import formulas
from .analytics import PerformanceAnalytics
from .comparisons import aligned
from .drawdowns import episodes
from .settings import AnalysisSettings

# The market's fall that makes a crisis unless told otherwise.
THRESHOLD = 0.15

# The months after a crisis's trough the returns are given for.
AFTER_MONTHS = (3, 6)


@dataclass(frozen=True)
class Episode:
    """A crisis window: a name, its start (the market's peak) and its end
    (its trough)."""
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


# Named episodes, S&P 500 closing peak to closing trough.
NAMED_CRISES = (
    Episode("Dot-com bust", pd.Timestamp("2000-03-24"), pd.Timestamp("2002-10-09")),
    Episode("Global financial crisis", pd.Timestamp("2007-10-09"), pd.Timestamp("2009-03-09")),
    Episode("European debt crisis", pd.Timestamp("2011-04-29"), pd.Timestamp("2011-10-03")),
    Episode("China and oil, 2015-16", pd.Timestamp("2015-05-21"), pd.Timestamp("2016-02-11")),
    Episode("Late-2018 sell-off", pd.Timestamp("2018-09-20"), pd.Timestamp("2018-12-24")),
    Episode("Covid crash", pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23")),
    Episode("2022 inflation bear market", pd.Timestamp("2022-01-03"),
            pd.Timestamp("2022-10-12")),
    Episode("2025 tariff shock", pd.Timestamp("2025-02-19"), pd.Timestamp("2025-04-08")),
)


@dataclass(frozen=True)
class Crisis:
    """How the fund held up in one crisis.

    Attributes:
        name: The episode's name, or the market's peak and trough dates.
        start: The window's start (the market's peak).
        end: Its end (the trough).
        partial: Whether the run covered only part of it.
        days: The days measured.
        fund: The fund's return over it.
        comparator: The comparator's.
        outperformance: The fund's less the comparator's.
        fund_drawdown: The fund's deepest fall inside it.
        comparator_drawdown: The comparator's.
        beta: The fund's beta to the comparator over it.
        alpha: The fund's return beyond what the whole run's beta predicts.
        after: The fund's and the comparator's returns over the months after
            the end, by ``fund 3m`` ... ``comparator 6m``; None where the run
            ends too soon.
        recovery_days: Trading days from the end until the fund was back at
            its value at the start; None if it had not been by the run's end.
    """
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    partial: bool
    days: int
    fund: float
    comparator: float
    outperformance: float
    fund_drawdown: float
    comparator_drawdown: float
    beta: float
    alpha: float
    after: dict[str, float | None]
    recovery_days: int | None


def market_crises(market: pd.Series,
                  threshold: float = THRESHOLD) -> list[Episode]:
    """Each fall of the market deeper than *threshold*, peak to trough."""
    return [Episode(f"{drawdown.peak.date()} to {drawdown.trough.date()}", drawdown.peak,
                    drawdown.trough)
            for drawdown in episodes(market.dropna().astype(float))
            if drawdown.depth <= -threshold]


def crisis_of(episode: Episode,
              ours: pd.Series,
              theirs: pd.Series,
              run_beta: float) -> Crisis | None:
    """How the fund held up over one episode, on the days the run has; None
    when the run has none of them."""
    inside = (ours.index > episode.start) & (ours.index <= episode.end)

    if not inside.any():
        return None

    fund, comparator = ours[inside], theirs[inside]
    total, theirs_total = formulas.compounded_return(fund), formulas.compounded_return(comparator)

    return Crisis(name=episode.name, start=episode.start, end=episode.end,
                  partial=bool(episode.start < ours.index[0] or episode.end > ours.index[-1]),
                  days=int(inside.sum()),
                  fund=total, comparator=theirs_total, outperformance=total - theirs_total,
                  fund_drawdown=formulas.max_drawdown(formulas.path_of(fund)),
                  comparator_drawdown=formulas.max_drawdown(formulas.path_of(comparator)),
                  beta=formulas.beta(fund, comparator),
                  alpha=total - run_beta * theirs_total,
                  after=_after(ours, theirs, episode.end),
                  recovery_days=_recovery(ours, episode))


def across(crises: Sequence[Crisis],
           ours: pd.Series,
           theirs: pd.Series,
           run_beta: float,
           periods_per_year: int) -> dict[str, float | None]:
    """The figures across every crisis: the average outperformance, the share
    outperformed, crisis capture, and the pooled crisis alpha (the mean daily
    return beyond the whole run's beta over every crisis day, a year)."""
    if not crises:
        return dict.fromkeys(("average_outperformance", "share_outperformed", "capture",
                              "pooled_alpha"))

    count = len(crises)
    fund_mean = float(np.prod([1 + crisis.fund for crisis in crises]) ** (1 / count) - 1)
    theirs_mean = float(np.prod([1 + crisis.comparator for crisis in crises]) ** (1 / count) - 1)
    days = np.zeros(len(ours), dtype=bool)

    for crisis in crises:
        days |= (ours.index > crisis.start) & (ours.index <= crisis.end)

    beyond = ours[days] - run_beta * theirs[days]

    return {"average_outperformance": float(np.mean([crisis.outperformance
                                                     for crisis in crises])),
            "share_outperformed": sum(crisis.outperformance > 0 for crisis in crises) / count,
            "capture": fund_mean / theirs_mean if theirs_mean != 0 else None,
            "pooled_alpha": float(beyond.mean() * periods_per_year)}


class CrisisAnalytics:
    """How a run held up in crises, against one comparator.

    Args:
        run: The run's data.
        comparator: What the fund is measured against in each crisis, by
            name or levels; the run's index by default.
        market: The broad market series the crises are found on, by name of
            one of the run's comparators or by its levels; the run's
            benchmark when it has one, else its index.
        threshold: The market's fall that makes a crisis.
        named: The user's own episodes, used instead of the market's falls.
        settings: What the figures are measured over and with.

    Raises:
        KeyError: If the run has no comparator of a name given.
    """

    def __init__(self,
                 run: RunData,
                 comparator: str | pd.Series = INDEX,
                 market: str | pd.Series | None = None,
                 threshold: float = THRESHOLD,
                 named: Sequence[Episode] | None = None,
                 settings: AnalysisSettings | None = None):
        performance = PerformanceAnalytics(run, settings)
        self.periods_per_year = performance.periods_per_year
        levels = run.comparators[comparator] if isinstance(comparator, str) else comparator
        self.ours, self.theirs = aligned(performance.returns, levels, performance.before)
        chosen = market if market is not None else (BENCHMARK if BENCHMARK in run.comparators
                                                   else INDEX)
        self.market = run.comparators[chosen] if isinstance(chosen, str) else chosen
        self.episodes = list(named) if named is not None else market_crises(self.market,
                                                                            threshold)
        self.run_beta = formulas.beta(self.ours, self.theirs)

    def crises(self) -> list[Crisis]:
        """Each crisis the run saw any of, in order."""
        found = (crisis_of(episode, self.ours, self.theirs, self.run_beta)
                 for episode in sorted(self.episodes, key=lambda episode: episode.start))

        return [crisis for crisis in found if crisis is not None]

    def summary(self) -> dict[str, float | None]:
        """The figures across every crisis."""
        return across(self.crises(), self.ours, self.theirs, self.run_beta,
                      self.periods_per_year)


def _after(ours: pd.Series,
           theirs: pd.Series,
           end: pd.Timestamp) -> dict[str, float | None]:
    """The fund's and comparator's returns over the months after *end*; None
    where the run ends before the months are out."""
    found: dict[str, float | None] = {}

    for months in AFTER_MONTHS:
        until = end + pd.DateOffset(months=months)
        window = (ours.index > end) & (ours.index <= until)
        complete = len(ours) > 0 and ours.index[-1] >= until

        found[f"fund {months}m"] = (formulas.compounded_return(ours[window])
                                    if complete else None)
        found[f"comparator {months}m"] = (formulas.compounded_return(theirs[window])
                                          if complete else None)

    return found


def _recovery(ours: pd.Series,
              episode: Episode) -> int | None:
    """Trading days from the episode's end until the fund was back at its
    value at the start; None if not by the run's end."""
    level = (1.0 + ours).cumprod()
    before = level[level.index <= episode.start]
    start_value = float(before.iloc[-1]) if len(before) else 1.0
    after = level[level.index > episode.end]
    back = np.flatnonzero(after.to_numpy() >= start_value)

    return int(back[0]) + 1 if len(back) else None
