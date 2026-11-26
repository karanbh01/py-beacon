# src/beacon/analysis/analytics/profile/factors.py
"""A name's profile's factors: the risk model's exposures of it, and its
returns' betas to the factors, at each rebalance."""
# BN-471 (Karan, 2026-11-25): both, side by side, so a disagreement can be
# read; a name's betas are noisier than a portfolio's, so their t-statistics
# are given with them.
import pandas as pd

from ...performance import formulas
from ...reports.statistics import TOO_SHORT, Unavailable
from ...risk import risk_store
from ...risk.factors import ALPHA, regressed


def factor_exposures(name: str,
                     table: pd.DataFrame,
                     days: pd.DatetimeIndex) -> pd.DataFrame:
    """At each rebalance over *days* in *table*
    (`risk_store.name_exposures`): the risk model's exposure of *name* to
    each factor, and in a global model its country, sector and currency;
    blank where the model left it out.

    Raises:
        Unavailable: If no rebalance over *days* had its model fitted.
    """
    table = table.loc[(table.index >= days[0]) & (table.index <= days[-1])]

    if table.empty:
        raise Unavailable(TOO_SHORT)

    dates = pd.DatetimeIndex(sorted(table.index.unique()))
    mine = table.loc[table[risk_store.IDENTIFIER] == name].drop(columns=risk_store.IDENTIFIER)
    found = mine.reindex(dates)
    kept = [str(column) for column in found.columns
            if str(column).startswith(risk_store.EXPOSURE) or found[column].notna().any()]
    # Labelled alike whether read back from a stored run or worked out.
    found = found[kept].set_axis(pd.Index(kept, dtype=object), axis=1)

    return found.rename_axis("DATE")


def factor_betas(name: str,
                 name_returns: pd.Series,
                 factor_returns: pd.DataFrame,
                 rebalances: list[pd.Timestamp],
                 window: int,
                 risk_free_rate: float,
                 periods_per_year: int) -> pd.DataFrame:
    """At each of *rebalances*: *name*'s daily returns over the risk-free
    rate regressed on the factor returns over the *window* days to it, each
    factor's beta and t-statistic, the regression's R-squared and the days
    it was fitted on; a rebalance with fewer days before it than *window*
    is left out.

    Raises:
        Unavailable: If no rebalance has *window* days before it.
    """
    rate = formulas.daily_rate(risk_free_rate, periods_per_year)
    paired = factor_returns.join((name_returns - rate).rename(ALPHA), how="inner").dropna()
    rows = {}

    for day in rebalances:
        before = paired.loc[paired.index <= day].iloc[-window:]

        if len(before) < window:
            continue

        fitted = regressed(before[ALPHA], before.drop(columns=ALPHA), periods_per_year)
        coefficients = fitted.coefficients.drop(ALPHA)
        rows[day] = {**{f"beta/{factor}": value
                        for factor, value in coefficients["coefficient"].items()},
                     **{f"t_stat/{factor}": value
                        for factor, value in coefficients["t_stat"].items()},
                     "r_squared": fitted.r_squared, "observations": fitted.observations}

    if not rows:
        raise Unavailable(TOO_SHORT)

    found = pd.DataFrame.from_dict(rows, orient="index")
    found.attrs = {"window": window}

    return found.rename_axis("DATE")
