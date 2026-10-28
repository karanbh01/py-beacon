# src/beacon/backtest/analytics_access.py
"""
A backtest result's analytics, each built on first read and kept: its
performance, portfolio, risk and attribution, with the settings of the
analysis the run was asked for, or the defaults.
"""
# BN-397. The analytics modules read a RunData, whose module imports the
# backtest result, so each is imported when first asked for.
from typing import Any, cast

from ..data.fetcher import DataFetcher


class AnalyticsMixin:
    """The result's analytics, mixed into BacktestResult."""

    analysis_request: Any
    _data_fetcher: DataFetcher | None

    def _kept(self,
              name: str,
              build: Any) -> Any:
        """The analytics kept under *name*, built by *build* the first time."""
        kept: dict[str, Any] = self.__dict__.setdefault("_analytics", {})

        if name not in kept:
            kept[name] = build()

        return kept[name]

    @property
    def run_data(self) -> Any:
        """The run as `beacon.analysis.RunData`, which every analysis reads."""
        from ..analysis.data import RunData  # noqa: PLC0415

        # Mixed into BacktestResult, which is what this always is.
        return self._kept("run_data", lambda: RunData.from_result(cast(Any, self)))

    @property
    def analytics(self) -> Any:
        """The run's performance (`beacon.analysis.performance.PerformanceAnalytics`)."""
        from ..analysis.performance.analytics import PerformanceAnalytics  # noqa: PLC0415

        return self._kept("analytics", lambda: PerformanceAnalytics(self.run_data,
                                                                    self._settings()))

    @property
    def portfolio_analytics(self) -> Any:
        """The run's portfolio figures (`beacon.analysis.performance.portfolio
        .PortfolioAnalytics`)."""
        from ..analysis.performance.portfolio import PortfolioAnalytics  # noqa: PLC0415

        return self._kept("portfolio_analytics", lambda: PortfolioAnalytics(self.run_data))

    @property
    def risk(self) -> Any:
        """The run's risk (`beacon.analysis.risk.analytics.RiskAnalytics`).

        Raises:
            ValueError: If the run kept no data source to measure from.
        """
        from ..analysis.risk.analytics import RiskAnalytics  # noqa: PLC0415

        source = self._data_fetcher

        if source is None:
            raise ValueError("The run kept no data source, which its risk is measured from.")

        request = self.analysis_request

        return self._kept("risk", lambda: RiskAnalytics(
            self.run_data, source, request.factors if request is not None else None,
            self._settings()))

    @property
    def attribution(self) -> Any:
        """The run's attribution (`beacon.analysis.attribution.Attribution`)."""
        from ..analysis.attribution.analytics import Attribution  # noqa: PLC0415

        return self._kept("attribution", lambda: Attribution(self.run_data))

    def _settings(self) -> Any:
        """The analysis settings asked for, or None for the defaults."""
        request = self.analysis_request

        return request.settings if request is not None else None
