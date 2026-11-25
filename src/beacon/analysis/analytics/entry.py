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
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Any

from ...data.fetcher import DataFetcher
from ..data import RunData
from ..reports.context import Context
from ..request import AnalysisRequest, analyse_run
from .portfolio import PortfolioLevel


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
    """

    def __init__(self,
                 run: RunData | Callable[[], RunData],
                 request: AnalysisRequest | None = None,
                 source: DataFetcher | None = None,
                 folder: Path | None = None,
                 block: dict[str, Any] | None = None,
                 timings: dict[str, float] | None = None,
                 context: Context | None = None):
        self._run = run
        self._request = request
        self._source = source
        self._folder = folder
        self._block = block
        self._timings = dict(timings or {})
        self._given_context = context

    @classmethod
    def of_result(cls,
                  result: Any) -> "Analytics":
        """The analytics of a finished backtest, `result`."""
        return cls(lambda: result.run_data, result.analysis_request,
                   getattr(result, "_data_fetcher", None), block=result.analysis_block,
                   timings=result.analysis_timings,
                   context=result.__dict__.get("_analysis_context"))

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

    @cached_property
    def portfolio(self) -> PortfolioLevel:
        """A whole portfolio's analytics, in nine categories: performance,
        risk, relative, holdings, trading, flows, income, vehicle and
        attribution."""
        return PortfolioLevel(self)

    @cached_property
    def _context(self) -> Context:
        """What the statistics are worked out from, shared by every
        category."""
        if self._given_context is not None:
            return self._given_context

        request = self._request or AnalysisRequest()

        return Context(self.run_data, request.options, self._source, request.factors,
                       self._folder)

    def __repr__(self) -> str:
        return "Analytics(portfolio)"
