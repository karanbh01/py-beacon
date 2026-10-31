# src/beacon/analysis/reports/catalogue/risk.py
"""
The statistics of the fund's risk: its factors, its value at risk, where its
risk comes from, how concentrated it is, and how fast it could be sold.

The factor, expected-risk and liquidity figures read the data store the run
was made from; without it they are unavailable.
"""
# BN-398/BN-400.
from collections.abc import Callable
from functools import partial
from typing import Any

import pandas as pd

from ....exceptions import CalculationError
from ...risk.var import ACTIVE
from ..context import FUND, Context
from ..statistics import (
    FIGURES,
    MODERATE,
    RISK,
    SERIES,
    TABLE,
    TOO_SHORT,
    Statistic,
    Summarised,
    Unavailable,
)
from .build import frame

FACTORS, LOSS, CONTRIBUTIONS = "factors", "value_at_risk", "risk_contributions"
CONCENTRATION, LIQUIDITY = "concentration", "liquidity"

VAR, ES = "var", "es"
LOSS_SETTINGS = ("value_at_risk.levels", "value_at_risk.horizons")
MODEL = ("risk_model.window", "risk_model.structure", "risk_model.country_field",
         "risk_model.minimum_names", "risk_model.sectors", "risk_model.currencies",
         "risk_model.weekly", "risk_model.weeks")


def long_enough(read: Callable[[Context], Any]) -> Callable[[Context], Any]:
    """*read*, or `Unavailable` when the run is too short for the risk model
    it needs."""
    def compute(context: Context) -> Any:
        try:
            return read(context)
        except CalculationError as error:
            raise Unavailable(TOO_SHORT, str(error)) from error

    return compute


def _exposures(context: Context) -> pd.DataFrame:
    found = context.factor_analytics
    parts = {FUND: found.holdings_exposures()}

    if context.tracked is not None:
        parts[ACTIVE] = found.active_exposures(context.tracked)

    return frame(pd.concat(parts, axis=1))


def _betas(context: Context) -> Summarised:
    found = context.factor_analytics.regression()

    return Summarised(found.coefficients, {"r_squared": found.r_squared,
                                           "observations": found.observations})


def _rolling_betas(context: Context) -> pd.DataFrame:
    return context.factor_analytics.rolling_betas(window=context.options.rolling.window)


def _loss(measure: str) -> Callable[[Context], dict[str, Any]]:
    """Value at risk or expected shortfall, by book, method, level and
    horizon."""
    def compute(context: Context) -> dict[str, Any]:
        options = context.options.value_at_risk
        found: dict[str, Any] = {}

        for book in context.loss.books():
            if book != FUND and book != ACTIVE:
                continue

            table = context.kept(("loss", book), partial(context.loss.table, book,
                                                         options.levels, options.horizons))
            rows = table.xs(measure, level="measure")

            for method, values in rows.iterrows():
                for (level, horizon), value in values.items():
                    found[f"{book}/{method}/{level}/{horizon}"] = value

        return found

    return compute


def _expected(columns: list[str]) -> Callable[[Context], pd.DataFrame]:
    """Columns of the expected risk at each rebalance."""
    def compute(context: Context) -> pd.DataFrame:
        against = context.tracked or "index"
        summary = context.kept(("expected", against),
                               lambda: context.risk_over_time.summary(against=against))

        return summary[[column for column in columns if column in summary.columns]]

    return compute


def _last(dates: list[pd.Timestamp]) -> pd.Timestamp:
    """The last of *dates*, or `Unavailable` when the run has none."""
    if not dates:
        raise Unavailable(TOO_SHORT, "The run has no rebalance with the history this needs.")

    return dates[-1]


def _contributions(context: Context) -> Summarised:
    over_time = context.risk_over_time
    day = _last(over_time.dates())

    return Summarised(over_time.contributions(day, against=context.tracked),
                      {"date": day})


def _by_weight(columns: list[str],
               optional: bool = False) -> Callable[[Context], pd.DataFrame]:
    def compute(context: Context) -> pd.DataFrame:
        table = context.kept(("by_weight", optional),
                             lambda: context.concentration.by_weight(optional=optional))

        return table[[column for column in columns if column in table.columns]]

    return compute


def _curve(context: Context) -> Summarised:
    day = _last(context.concentration.rebalances())

    return Summarised(context.concentration.curve(day), {"date": day})


def _absorption(context: Context) -> dict[str, Any]:
    # At the last rebalance only, as it was when each took an
    # eigendecomposition of the holdings' covariance (BN-418 made it quick).
    day = _last(context.concentration.rebalances())

    return {"ratio": context.concentration.absorption_ratio(day), "date": day}


def _days_to_sell(context: Context) -> pd.DataFrame:
    return context.liquidity.profile(horizons=context.options.liquidity.horizons)


def statistics() -> list[Statistic]:
    """The risk statistics, in order."""
    def stat(id_: str,
             label: str,
             category: str,
             kind: str,
             description: str,
             read: Callable[[Context], Any],
             **details: Any) -> Statistic:
        return Statistic(id_, label, category, kind, RISK, description, long_enough(read),
                         **details)

    expected = ["volatility", "factor_volatility", "specific_volatility", "factor_share",
                "tracking_error", "factor_tracking_error", "specific_tracking_error"]

    return [
        stat("factor_exposures", "Factor exposures", FACTORS, TABLE, "Each book's "
             "standardised exposure to the factors at each rebalance, and the active "
             "exposures.", _exposures, unit="ratio", settings=("window.tracked",)),
        stat("factor_returns", "Factor returns", FACTORS, TABLE, "Each day's return to each "
             "factor from a cap-weighted cross-sectional regression.",
             lambda context: context.factor_analytics.factor_returns(), unit="fraction"),
        stat("factor_betas", "Factor betas", FACTORS, TABLE, "Returns-based betas of the "
             "fund to the factors, with t-statistics, R-squared and alpha.", _betas,
             unit="ratio", rolling=MODERATE, roll=_rolling_betas,
             settings=("rolling.window",)),
        stat("value_at_risk", "Value at risk", LOSS, FIGURES, "The loss exceeded one day (or "
             "more) in twenty or a hundred: historical, normal and Cornish-Fisher.",
             _loss(VAR), unit="fraction", rolling=MODERATE, settings=LOSS_SETTINGS,
             figure_names=("book", "method", "confidence", "horizon")),
        stat("expected_shortfall", "Expected shortfall", LOSS, FIGURES, "The mean loss on the "
             "days worse than the value at risk: historical and normal.", _loss(ES),
             unit="fraction", rolling=MODERATE, settings=LOSS_SETTINGS,
             figure_names=("book", "method", "confidence", "horizon")),
        stat("stress_windows", "Stress windows", LOSS, TABLE, "Each book's return and "
             "deepest fall in each crisis window.",
             lambda context: frame(context.loss.stress_windows(
                 context.market(), context.options.crises.threshold)),
             unit="fraction", settings=("crises.market", "crises.threshold")),
        stat("expected_risk", "Expected volatility and tracking error", CONTRIBUTIONS, TABLE,
             "From the factor model at each rebalance, split into factor and specific.",
             _expected(expected), unit="fraction", settings=MODEL),
        stat("risk_contributions", "Risk contributions", CONTRIBUTIONS, TABLE, "Each name's "
             "marginal contribution, contribution and share of expected risk at the last "
             "rebalance.", _contributions, settings=MODEL),
        stat("measured_contributions", "Measured contributions", CONTRIBUTIONS, TABLE,
             "Each name's share of the fund's measured variance over the run.",
             lambda context: context.risk_over_time.measured()),
        stat("diversification_ratio", "Diversification ratio", CONTRIBUTIONS, TABLE,
             "The weighted sum of names' volatilities over the book's, at each rebalance.",
             _expected(["diversification_ratio"]), unit="ratio", settings=MODEL),
        stat("average_correlation", "Average correlation", CONTRIBUTIONS, TABLE,
             "The correlation between holdings the weights and volatilities imply, at each "
             "rebalance.", _expected(["average_correlation"]), unit="ratio", settings=MODEL),
        stat("herfindahl", "Herfindahl and effective number", CONCENTRATION, TABLE,
             "Herfindahl of weights, its normalised form, and the effective number of "
             "holdings, at each rebalance.",
             _by_weight(["names", "herfindahl", "normalised_herfindahl", "effective_number"])),
        stat("top_weights", "Top weights and coverage", CONCENTRATION, TABLE, "Largest weight, "
             "top 5, 10, 20, 25, and the fewest names making 50%, 80%, 90%, at each "
             "rebalance.", _by_weight(["largest", "top_5", "top_10", "top_20", "top_25",
                                       "names_for_50", "names_for_80", "names_for_90"])),
        stat("concentration_curve", "Concentration curve", CONCENTRATION, TABLE, "Names "
             "largest first at the last rebalance: each one's weight, the cumulative weight "
             "to it, and how the top k's concentration changes as each is added.", _curve),
        stat("gini_and_entropy", "Gini and entropy", CONCENTRATION, TABLE, "Gini coefficient "
             "from the Lorenz curve, and entropy with its effective number, at each "
             "rebalance.", _by_weight(["gini", "entropy", "entropy_effective_number"],
                                      optional=True)),
        stat("concentration_by_risk", "Concentration by risk", CONCENTRATION, TABLE,
             "Herfindahl of risk shares and effective number of risk sources; top names' "
             "share of risk.", lambda context: context.concentration.by_risk(),
             settings=MODEL),
        stat("absorption_ratio", "Absorption ratio", CONCENTRATION, FIGURES, "Share of the "
             "held names' total variance in the risk model's largest eigenvalues, one per "
             "independent factor, at the last rebalance.", _absorption, unit="fraction",
             settings=MODEL,
             figure_names=("field",)),
        stat("concentration_by_outcome", "Concentration by outcome", CONCENTRATION, FIGURES,
             "Top contributors' share of the return, and breadth.",
             lambda context: context.concentration.by_outcome(*context.window()),
             unit="fraction", figure_names=("field",)),
        stat("drift", "Drift", CONCENTRATION, SERIES, "The share of the book that would trade "
             "to return to the last rebalance's weights, each day.",
             lambda context: context.concentration.drift(), unit="fraction"),
        stat("days_to_sell", "Days to sell and sellable share", LIQUIDITY, TABLE, "Days to "
             "sell the book at its participation in daily traded value, and the share "
             "sellable in 1, 5 and 20 days.", _days_to_sell,
             settings=("liquidity.participation", "liquidity.lookback_days",
                       "liquidity.horizons")),
    ]
