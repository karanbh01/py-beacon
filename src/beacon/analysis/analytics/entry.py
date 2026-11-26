# src/beacon/analysis/analytics/entry.py
"""
A run's analytics, the one way in to every calculation on it:

    result.analytics.portfolio.performance   # a whole portfolio's figures
    result.analytics.request                 # the analysis the run was asked for
    result.analytics.to_dict()               # its analytics block

Nothing is worked out until it is asked for, and each answer is kept.
"""
# BN-433 (Karan, 2026-11-24): replaces 0.10.0's `result.analytics`, which was
# the fund's PerformanceAnalytics.
from collections.abc import Callable, Iterable, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from ...data.fetcher import DataFetcher
from ..data import RunData
from ..reports.catalogue import CATALOGUE
from ..reports.context import Context
from ..reports.options import RollingOptions
from ..request import AnalysisRequest, analyse_run
from .forms import value, windowed
from .instrument import InstrumentLevel
from .portfolio import PortfolioLevel
from .reports import Reports
from .shape import ALL
from .tables import table as side_by_side


class Analytics:
    """A run's analytics.

    Args:
        run: The run, or what gives it when first read.
        request: The analysis the run was asked for; None when it was asked
            for none, and the defaults are used.
        source: The data store the run was made from, which factor,
            liquidity and risk figures read; None when it is not loaded.
        folder: Where a stored run is kept, whose factor work is read from
            there.
        block: The analytics block already worked out for *request*.
        timings: How long each part of that block took, in seconds.
        context: What the run's statistics were worked out from, when they
            were, so its work is shared.
        active: An active strategy's steps, one each rebalance, as a
            finished run holds them; None when they are not kept.
        replication: An index tracking strategy's replication steps, one
            each rebalance, as a finished run holds them; None when they are
            not kept.
    """

    def __init__(self,
                 run: RunData | Callable[[], RunData],
                 request: AnalysisRequest | None = None,
                 source: DataFetcher | None = None,
                 folder: Path | None = None,
                 block: dict[str, Any] | None = None,
                 timings: dict[str, float] | None = None,
                 context: Context | None = None,
                 active: Sequence[Any] | None = None,
                 replication: Sequence[Any] | None = None):
        self._run = run
        self._request = request
        self._source = source
        self._folder = folder
        self._block = block
        self._timings = dict(timings or {})
        self._given_context = context
        self._contexts: dict[tuple[Any, ...], Context] = {}
        self._active = active
        self._replication = replication

    @classmethod
    def of_result(cls,
                  result: Any) -> "Analytics":
        """The analytics of a finished backtest, `result`."""
        return cls(lambda: result._run_data, result._analysis_request,
                   getattr(result, "_data_fetcher", None), block=result._analysis_block,
                   timings=result._analysis_timings,
                   context=result.__dict__.get("_analysis_context"),
                   active=result.trading.active or [],
                   replication=result.trading.replication or [])

    @cached_property
    def run_data(self) -> RunData:
        """The run as `beacon.analysis.RunData`, which every figure reads."""
        return self._run if isinstance(self._run, RunData) else self._run()

    @property
    def request(self) -> AnalysisRequest | None:
        """The analysis the run was asked for; None when it was asked for
        none."""
        return self._request

    @property
    def timings(self) -> dict[str, float]:
        """How long each part of the run's analysis took, in seconds, the
        run's tables first; empty until it is worked out."""
        return dict(self._timings)

    def to_dict(self) -> dict[str, Any]:
        """The run's analytics block, as the API gives it: the analysis it
        was asked for, or the default factsheet, worked out the first time.

        Returns:
            dict: As `beacon.analysis.request.analyse` gives it.
        """
        if self._block is None:
            self._block = analyse_run(self._context, self._request or AnalysisRequest())
            self._timings = self._timings or dict(self._block["timings"])

        return self._block

    def get(self,
            statistic: str,
            **form: Any) -> Any:
        """Any statistic by its id or `Statistic` member, with the keywords
        its method takes: ``get(Statistic.TOTAL_RETURN, by="year")`` is
        ``portfolio.performance.total_return(by="year")``.

        Raises:
            ValueError: If no statistic has that id.
            TypeError: If a keyword is not one the statistic takes.
            Unavailable: If the run cannot give it.
        """
        if str(statistic) not in CATALOGUE:
            raise ValueError(f"Unknown statistic {str(statistic)!r}.")

        return self._value(str(statistic), form)

    def table(self,
              statistics: Iterable[str],
              **form: Any) -> pd.DataFrame:
        """Several single-figure statistics side by side, by id or
        `Statistic` member: a row a statistic, a column a window, period or
        portfolio (`beacon.analysis.analytics.tables`).

            analytics.table([Statistic.TOTAL_RETURN, Statistic.SHARPE_RATIO],
                            trailing=["1Y", "3Y"], of="all")

        Raises:
            ValueError: If a statistic gives more than one figure.
            TypeError: If a keyword is one no table takes, or none of the
                statistics takes.
        """
        return side_by_side(statistics, form, self._value)

    @cached_property
    def portfolio(self) -> PortfolioLevel:
        """A whole portfolio's analytics, in nine categories: performance,
        risk, relative, holdings, trading, flows, income, vehicle and
        attribution."""
        return PortfolioLevel(self)

    @cached_property
    def reports(self) -> Reports:
        """The run's reports, a method a report, each its statistics worked
        out together, and those its request asked for (``asked``)."""
        return Reports(self)

    def _with(self,
              request: AnalysisRequest) -> "Analytics":
        """These analytics under another request's choices."""
        return Analytics(self._run, request, self._source, self._folder,
                         active=self._active, replication=self._replication)

    @cached_property
    def instrument(self) -> InstrumentLevel:
        """The analytics a name at a time: each name's return, risk and
        figures against a comparator, and the statistics given a row a
        name."""
        return InstrumentLevel(self)

    @cached_property
    def _context(self) -> Context:
        """What the statistics are worked out from, shared by every
        category."""
        if self._given_context is not None:
            return self._given_context

        request = self._request or AnalysisRequest()

        return Context(self.run_data, request.options, self._source, request.factors,
                       self._folder)

    def _value(self,
               statistic_id: str,
               form: dict[str, Any]) -> Any:
        """Statistic *statistic_id* in the form its keywords *form* ask for
        (`beacon.analysis.analytics.forms`)."""
        return value(CATALOGUE[statistic_id], form, self._context_for)

    def _returns(self,
                 of: str,
                 start: Any = None,
                 end: Any = None) -> Any:
        """Daily returns over a window: portfolio *of*'s, or each one's in a
        column (``all``).

        Raises:
            ValueError: If the run has no portfolio *of*.
        """
        context = self._context_for(start=start, end=end)
        books = context.books()

        if of != ALL and of not in books:
            raise ValueError(f"No portfolio {of!r} here. Known: {', '.join(books)}.")

        if of != ALL:
            return context.book(of).returns

        return pd.DataFrame({name: context.book(name).returns for name in books})

    def _active_share(self) -> pd.Series:
        """An active strategy's active share at each rebalance; empty for
        a strategy that is not active.

        Raises:
            ValueError: If the run's active steps were not kept with it.
        """
        if self._active is None:
            raise ValueError("The run's active steps were not kept with it.")

        return pd.Series({step.date: step.active_share for step in self._active}, dtype=float)

    def _ex_ante_tracking_error(self) -> pd.Series:
        """The strategy's own forecast of its tracking error at each
        rebalance: an active strategy's, else an index tracking strategy's
        replication's, blank where it was not estimated.

        Raises:
            ValueError: If the run's rebalance steps were not kept with it.
        """
        # BN-451.
        if self._active is None and self._replication is None:
            raise ValueError("The run's rebalance steps were not kept with it.")

        steps = self._active or self._replication or []

        return pd.Series({step.date: (float("nan") if step.tracking_error is None
                                      else step.tracking_error) for step in steps},
                         dtype=float)

    def _context_for(self,
                     start: Any = None,
                     end: Any = None,
                     tracked: str | None = None,
                     rolling: RollingOptions | None = None,
                     annualise: bool | None = None) -> Context:
        """The context over a window, tracking a comparator, rolling a
        window or annualising another way, built once for each."""
        key = (start and str(start), end and str(end), tracked, rolling, annualise)

        if key not in self._contexts:
            self._contexts[key] = windowed(self._context, start, end, tracked, rolling,
                                           annualise)

        return self._contexts[key]

    def __repr__(self) -> str:
        return "Analytics(portfolio)"
