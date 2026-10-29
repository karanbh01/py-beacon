# src/beacon/analysis/reports/catalogue/performance.py
"""
The statistics of a run's own returns: returns, risk, drawdowns, the shape of
the distribution and the risk-adjusted ratios, each given for the fund, its
comparators and its share classes.
"""
# BN-398/BN-400.
from dataclasses import asdict
from typing import Any

import pandas as pd

from ...performance import formulas, ratios
from ...performance.analytics import PerformanceAnalytics
from ..context import Context
from ..statistics import (
    FIGURES,
    LIGHT,
    MODERATE,
    NO_FLOWS,
    PERFORMANCE,
    SERIES,
    TABLE,
    Column,
    Statistic,
    Summarised,
    Unavailable,
)
from .build import frame as wire_frame
from .build import fund_only, memo, per_book

RETURNS, RISK, DRAWDOWN = "returns", "risk", "drawdown"
DISTRIBUTION, RISK_ADJUSTED = "distribution", "risk_adjusted"


def _headline(name: str) -> Any:
    return lambda book, context: memo(book, "headline", book.headline)[name]


def _excess(book: PerformanceAnalytics) -> pd.Series:
    return memo(book, "excess", lambda: ratios.excess_returns(book.returns, book.periods_per_year,
                                                              book.risk_free_rate))


# The ratios that read only the excess returns, each worked out on its own, as
# `PerformanceAnalytics.ratios` works it out, so a rolling one does not work
# out the drawdowns the others need.
ALONE: dict[str, Any] = {
    "sortino": lambda book, options: ratios.sortino(_excess(book), book.periods_per_year),
    "kappa_3": lambda book, options: ratios.kappa_3(_excess(book)),
    "upside_potential": lambda book, options: ratios.upside_potential(_excess(book)),
    "omega": lambda book, options: ratios.omega(_excess(book)),
    "gain_to_pain": lambda book, options: ratios.gain_to_pain(
        formulas.period_returns(book.returns, formulas.MONTHLY)),
    "probabilistic_sharpe": lambda book, options: ratios.probabilistic_sharpe(
        _excess(book), book.periods_per_year, options.sharpe_benchmark),
    "deflated_sharpe": lambda book, options: ratios.deflated_sharpe(
        _excess(book), book.periods_per_year, options.trials, options.trial_variance),
    "lo_sharpe": lambda book, options: ratios.lo_sharpe(_excess(book), book.periods_per_year),
}


def _ratio(name: str) -> Any:
    def read(book: PerformanceAnalytics,
             context: Context) -> Any:
        options = context.options.returns

        if name in ALONE:
            return ALONE[name](book, options)

        return memo(book, "ratios", lambda: book.ratios(options.sharpe_benchmark, options.trials,
                                                        options.trial_variance))[name]

    return read


def _shape(name: str) -> Any:
    def read(book: PerformanceAnalytics,
             context: Context) -> Any:
        return memo(book, "distribution",
                    lambda: book.distribution(context.options.returns.histogram_bins))[name]

    return read


def _money_weighted(book: PerformanceAnalytics,
                    context: Context) -> Any:
    if context.run.investor_flows is None:
        raise Unavailable(NO_FLOWS)

    return book.money_weighted_return()


def _periods(context: Context) -> pd.DataFrame:
    """Each month's, quarter's and year's figures, in one table."""
    tables = []

    for frequency in (formulas.MONTHLY, formulas.QUARTERLY, formulas.ANNUAL):
        table = wire_frame(context.performance.period_table(frequency))
        table.insert(0, "frequency", frequency)
        tables.append(table)

    found = pd.concat(tables)
    found.index = found.index.astype(str)

    return found


def _best_and_worst(book: PerformanceAnalytics,
                    context: Context) -> dict[str, Any]:
    return {frequency: {"best": pair[0], "worst": pair[1]}
            for frequency, pair in book.best_and_worst().items() if pair is not None}


def _streaks(book: PerformanceAnalytics,
             context: Context) -> dict[str, Any]:
    winning, losing = book.longest_streaks()

    return {"winning": winning, "losing": losing}


def _episodes(context: Context) -> Summarised:
    """The deepest drawdowns, and what sums the run's drawdowns up."""
    found = context.performance.drawdowns(context.options.returns.worst_drawdowns)
    table = pd.DataFrame([asdict(drawdown) for drawdown in found.worst],
                         columns=[column.name for column in EPISODE_COLUMNS])
    summary = {name: value for name, value in asdict(found).items()
               if name not in ("worst", "current_drawdown")}

    return Summarised(table, summary)


def _histogram(context: Context) -> pd.DataFrame:
    histogram = _shape("histogram")(context.performance, context)
    edges = histogram.edges

    return pd.DataFrame({"low": edges[:-1], "high": edges[1:], "count": histogram.counts,
                         "normal": histogram.normal})


def _bootstrap(context: Context) -> dict[str, Any]:
    options = context.options.returns
    ranges = context.performance.bootstrap(options.bootstrap_samples, options.bootstrap_confidence,
                                           options.bootstrap_block, options.bootstrap_seed)

    return dict(ranges or {})


EPISODE_COLUMNS = (Column("peak", "Peak", "date"), Column("trough", "Trough", "date"),
                   Column("recovery", "Recovery", "date"), Column("depth", "Depth", "fraction"),
                   Column("to_trough", "To the trough", "days"),
                   Column("to_recover", "To recover", "days"), Column("length", "Length", "days"))

RATIO_SETTINGS = ("window.risk_free_rate",)


def statistics() -> list[Statistic]:
    """The statistics of a run's own returns, in order."""
    return [
        Statistic("total_return", "Total return", RETURNS, FIGURES, PERFORMANCE,
                  "The fund's compounded return over the window.",
                  per_book(_headline("total_return"), active=True), unit="fraction",
                  rolling=LIGHT),
        Statistic("annualised_return", "Annualised return (CAGR)", RETURNS, FIGURES, PERFORMANCE,
                  "The constant yearly rate that compounds to the total return.",
                  per_book(_headline("annualised_return"), active=True), unit="fraction",
                  rolling=LIGHT, settings=("window.periods_per_year",)),
        Statistic("money_weighted_return", "Money-weighted return", RETURNS, FIGURES,
                  PERFORMANCE, "The return an investor earned given when money came in and "
                  "went out (the IRR).", fund_only(_money_weighted), unit="fraction"),
        Statistic("trailing_returns", "Trailing returns", RETURNS, TABLE, PERFORMANCE,
                  "The return over the last 1, 3 and 6 months, 1, 3, 5 and 10 years, month, "
                  "quarter and year to date, and since inception; annualised beyond a year.",
                  lambda context: wire_frame(context.performance.trailing_table()),
                  unit="fraction"),
        Statistic("period_returns", "Period returns", RETURNS, TABLE, PERFORMANCE,
                  "Each month's, quarter's and year's return, against the comparators'.",
                  _periods, unit="fraction"),
        Statistic("best_and_worst", "Best and worst periods", RETURNS, FIGURES, PERFORMANCE,
                  "The best and worst day, week, month, quarter and year.",
                  per_book(_best_and_worst), unit="fraction",
                  figure_names=("book", "frequency", "extreme", "field")),
        Statistic("positive_share", "Share of positive periods", RETURNS, FIGURES, PERFORMANCE,
                  "The share of days, months and years with a gain.",
                  per_book(lambda book, context: book.positive_shares()), unit="fraction",
                  rolling=LIGHT, figure_names=("book", "frequency")),
        Statistic("gains_and_losses", "Average gain and loss", RETURNS, FIGURES, PERFORMANCE,
                  "The mean of the periods that gained and of those that lost, and their "
                  "ratio.", per_book(lambda book, context: book.gains_and_losses(
                      context.options.returns.gains_frequency)), unit="fraction",
                  rolling=LIGHT, settings=("returns.gains_frequency",),
                  figure_names=("book", "field")),
        Statistic("streaks", "Longest winning and losing streaks", RETURNS, FIGURES,
                  PERFORMANCE, "The most months in a row with a gain, and with a loss.",
                  per_book(_streaks), figure_names=("book", "streak", "field")),
        Statistic("mean_return", "Arithmetic and geometric mean return", RETURNS, FIGURES,
                  PERFORMANCE, "The mean daily return, and the daily rate that compounds to "
                  "the total, each also a year.", per_book(lambda book, context: book.means()),
                  unit="fraction", rolling=LIGHT, figure_names=("book", "mean")),
        Statistic("hit_rate", "Hit rate (monthly)", RETURNS, FIGURES, PERFORMANCE,
                  "The share of months the fund rose.",
                  per_book(lambda book, context: book.hit_rate_monthly()), unit="fraction",
                  rolling=LIGHT),
        Statistic("month_of_year", "Month of year", RETURNS, SERIES, PERFORMANCE,
                  "The mean return of each calendar month across the years.",
                  lambda context: context.performance.month_of_year(), unit="fraction"),
        Statistic("volatility", "Volatility", RISK, FIGURES, PERFORMANCE,
                  "The annualised standard deviation of daily returns.",
                  per_book(_headline("volatility")), unit="fraction", rolling=LIGHT),
        Statistic("max_drawdown", "Maximum drawdown", DRAWDOWN, FIGURES, PERFORMANCE,
                  "The deepest fall from a running peak.",
                  per_book(_headline("max_drawdown")), unit="fraction", rolling=MODERATE),
        Statistic("drawdown_episodes", "Drawdown episodes", DRAWDOWN, TABLE, PERFORMANCE,
                  "Each fall from a peak: its depth, start, trough, recovery, and how long "
                  "each part took.", _episodes, settings=("returns.worst_drawdowns",),
                  columns=EPISODE_COLUMNS),
        *_distribution(),
        *_ratios(),
    ]


def _distribution() -> list[Statistic]:
    def shape(id_: str,
              label: str,
              description: str,
              unit: str | None = None,
              names: tuple[str, ...] = ("book",)) -> Statistic:
        return Statistic(id_, label, DISTRIBUTION, FIGURES, PERFORMANCE, description,
                         per_book(_shape(id_)), unit=unit, rolling=LIGHT, figure_names=names)

    return [
        shape("downside_deviation", "Downside deviation", "The spread of returns below a "
              "target (the daily risk-free rate).", unit="fraction"),
        shape("semi_variance", "Semi-variance", "The variance of returns below their mean."),
        shape("skew", "Skew", "How lopsided returns are; negative means a longer left tail.",
              unit="ratio"),
        shape("excess_kurtosis", "Excess kurtosis", "How fat the tails are beyond a normal "
              "distribution's.", unit="ratio"),
        shape("tail_ratio", "Tail ratio", "The 95th percentile return over the size of the "
              "5th.", unit="ratio"),
        shape("serial_correlation", "Serial correlation", "The correlation of each day's "
              "return with the days before.", unit="ratio", names=("book", "lag")),
        Statistic("return_histogram", "Return histogram", DISTRIBUTION, TABLE, PERFORMANCE,
                  "Daily returns counted into bins, with what a normal distribution would put "
                  "in each.", _histogram, settings=("returns.histogram_bins",),
                  columns=(Column("low", "From", "fraction"), Column("high", "To", "fraction"),
                           Column("count", "Days", "count"),
                           Column("normal", "Normal", "count"))),
        Statistic("bootstrap_ranges", "Bootstrap ranges", DISTRIBUTION, FIGURES, PERFORMANCE,
                  "Confidence ranges for the main figures from resampled returns.",
                  _bootstrap, settings=("returns.bootstrap_samples",
                                        "returns.bootstrap_confidence", "returns.bootstrap_block",
                                        "returns.bootstrap_seed"),
                  figure_names=("figure", "field")),
    ]


def _ratios() -> list[Statistic]:
    def ratio(id_: str,
              key: str,
              label: str,
              description: str,
              rolling: str = LIGHT,
              settings: tuple[str, ...] = RATIO_SETTINGS,
              unit: str = "ratio") -> Statistic:
        return Statistic(id_, label, RISK_ADJUSTED, FIGURES, PERFORMANCE, description,
                         per_book(_ratio(key)), unit=unit, rolling=rolling,
                         settings=settings)

    return [
        Statistic("sharpe_ratio", "Sharpe ratio", RISK_ADJUSTED, FIGURES, PERFORMANCE,
                  "Mean daily return over the risk-free rate, per unit of its volatility, "
                  "annualised.", per_book(_headline("sharpe_ratio")), unit="ratio",
                  rolling=LIGHT, settings=RATIO_SETTINGS),
        ratio("sortino_ratio", "sortino", "Sortino ratio", "The Sharpe ratio with only "
              "downside deviation as risk."),
        ratio("calmar_ratio", "calmar", "Calmar ratio", "The annualised return over the "
              "deepest drawdown.", rolling=MODERATE),
        ratio("omega_ratio", "omega", "Omega ratio", "Gains above the daily risk-free rate "
              "over losses below it."),
        ratio("kappa_3", "kappa_3", "Kappa 3", "Excess return over the third lower partial "
              "moment."),
        ratio("upside_potential_ratio", "upside_potential", "Upside potential ratio",
              "Mean gain above the target over downside deviation."),
        ratio("martin_ratio", "martin", "Martin ratio (Ulcer)", "Excess annual return over "
              "the Ulcer index.", rolling=MODERATE),
        ratio("sterling_ratio", "sterling", "Sterling ratio", "Annual return over the average "
              "of the yearly maximum drawdowns.", rolling=MODERATE),
        ratio("burke_ratio", "burke", "Burke ratio", "Excess annual return over the root sum "
              "of squared drawdowns.", rolling=MODERATE),
        ratio("gain_to_pain", "gain_to_pain", "Gain to pain", "The sum of monthly returns "
              "over the sum of monthly losses."),
        ratio("probabilistic_sharpe_ratio", "probabilistic_sharpe", "Probabilistic Sharpe "
              "ratio", "The probability the true Sharpe ratio is above the one to beat, given "
              "the sample's length, skew and kurtosis.",
              settings=(*RATIO_SETTINGS, "returns.sharpe_benchmark"), unit="fraction"),
        ratio("deflated_sharpe_ratio", "deflated_sharpe", "Deflated Sharpe ratio", "The "
              "probabilistic Sharpe ratio against the best of many trials by chance.",
              settings=(*RATIO_SETTINGS, "returns.trials", "returns.trial_variance"),
              unit="fraction"),
        ratio("lo_sharpe_ratio", "lo_sharpe", "Lo-adjusted Sharpe ratio", "The Sharpe ratio "
              "corrected for serial correlation when annualising."),
    ]
