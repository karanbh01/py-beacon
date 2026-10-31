# src/beacon/analysis/risk/contributions.py
"""
Where a run's risk came from: expected at each rebalance, from the factor
model, and measured over the run, from its returns.

**Expected.** At each rebalance the factor model (see
`beacon.analysis.risk.model`) is estimated on the year of daily returns
before it, on the factors' exposures that day, and the book is held at its
weights at that day's close. Its covariance is ``B F B' + D``, so the
book's variance splits exactly into a factor part, ``x' F x`` with
``x = B' w``, and a specific part, ``w' D w``; so does the active variance,
on the active weights. Each name's marginal contribution is ``(S w)_i / sigma``
and its contribution ``w_i`` times that, which add up to the volatility;
a group's is the sum of its names'.

The diversification ratio is the weighted sum of the names' volatilities
over the book's, and the average correlation the one the weights and
volatilities imply:

    DR = sum w_i s_i / s,    rho = (s^2 - sum w_i^2 s_i^2) / ((sum w_i s_i)^2 - sum w_i^2 s_i^2)

**Measured.** Each name's daily contribution to the fund's return is its
weight at the close before times its return that day; whatever else moved the
fund (cash, costs, futures) is ``other``. Each one's share of the fund's
measured variance is the covariance of its contributions with the fund's
returns over their variance, and the shares add up to 1 exactly.
"""
# BN-389. Agreed with Karan on 2026-10-27.
from dataclasses import replace

import numpy as np
import pandas as pd

from ...data.base import as_of_position
from ...risk.model import FactorRiskModel
from ..panels import BENCHMARK_WEIGHTS, FUND_WEIGHTS, INDEX_WEIGHTS, NAME_RETURNS, REFERENCE
from .factors import FactorAnalytics
from .model import FACTOR, RebalanceRiskModels, RiskModelSettings

BOOKS = {"fund": FUND_WEIGHTS, "index": INDEX_WEIGHTS, "benchmark": BENCHMARK_WEIGHTS}
OTHER = "other"

# What each rebalance's summary gives.
SUMMARY = ("volatility", "factor_volatility", "specific_volatility", "factor_share",
           "tracking_error", "factor_tracking_error", "specific_tracking_error",
           "diversification_ratio", "average_correlation", "covered_weight")


class RiskOverTime:
    """A run's expected risk at each rebalance and its measured risk over
    the run.

    Args:
        factors: The run's factors, which the risk model is fitted on.
        window: The sessions of daily returns before each rebalance the
            model is estimated on.
        settings: How the factor model is estimated, its structure among
            them (see `beacon.analysis.risk.model`); its kind is always a
            factor model, and *window* is used when None.
    """

    def __init__(self,
                 factors: FactorAnalytics,
                 window: int = 252,
                 settings: RiskModelSettings | None = None):
        self.factors = factors
        self.run = factors.run
        chosen = replace(settings or RiskModelSettings(window=window), kind=FACTOR)
        self.models = RebalanceRiskModels(
            self.run, chosen,
            exposures=lambda day, names: factors.exposures(day).reindex(names),
            caps=lambda day, names: factors.caps(day).reindex(names).fillna(0.0))

    def dates(self) -> list[pd.Timestamp]:
        """The rebalances with enough returns before them for their model."""
        return [day for day in self.models.dates if self.models.has_history(day)]

    def weights(self,
                date: pd.Timestamp,
                book: str = "fund") -> pd.Series:
        """A book's weights at the close of *date*, by name."""
        table = self.run.tables.get(BOOKS[book])

        if table is None or table.empty:
            raise KeyError(f"The run kept no weights for its {book}.")

        at = as_of_position(table.index, date)
        row = table.iloc[at].fillna(0.0) if at is not None else pd.Series(dtype=float)
        row.index = row.index.astype(str)

        return row[row != 0.0]

    def contributions(self,
                      date: str | pd.Timestamp,
                      book: str = "fund",
                      against: str | None = None) -> pd.DataFrame:
        """Each name's weight, marginal contribution, contribution and share
        of the book's expected volatility (or, *against* a comparator, of its
        expected tracking error) at the rebalance on *date*."""
        day = pd.Timestamp(date)
        model = self._model(day)
        weights = self._vector(model, day, book, against)
        moved = _covariance_times(model, weights)
        sigma = float(np.sqrt(max(float(weights @ moved), 0.0)))
        marginal = moved / sigma if sigma > 0.0 else moved * 0.0
        contribution = weights * marginal
        table = pd.DataFrame({"weight": weights, "marginal": marginal,
                              "contribution": contribution,
                              "share": contribution / sigma if sigma > 0.0 else 0.0},
                             index=model.asset_ids)

        return table[table["weight"] != 0.0].sort_values("contribution", ascending=False)

    def group_contributions(self,
                            date: str | pd.Timestamp,
                            field: str,
                            book: str = "fund",
                            against: str | None = None) -> pd.DataFrame:
        """The contributions added up by the group each name was in on the
        day, by the grouping field *field*."""
        day = pd.Timestamp(date)
        table = self.contributions(day, book, against)
        groups = self._groups(day, field).reindex(table.index).fillna("Unknown")

        return (table[["weight", "contribution", "share"]].groupby(groups).sum()
                .sort_values("contribution", ascending=False))

    def expected(self,
                 date: str | pd.Timestamp,
                 book: str = "fund",
                 against: str = "index") -> dict[str, float]:
        """The book's expected risk at the rebalance on *date*: its
        volatility, split into factor and specific; its tracking error
        against *against*, split the same way; its diversification ratio and
        average correlation; and the share of its weight the model covers."""
        day = pd.Timestamp(date)
        model = self._model(day)
        weights = self._vector(model, day, book, None)
        held = self.weights(day, book)
        found = _split(model, weights, "")
        found.update(_split(model, self._vector(model, day, book, against), "tracking_"))
        found.update(_diversification(model, weights))
        found["covered_weight"] = float(weights.sum() / held.sum()) if held.sum() else 0.0

        return found

    def summary(self,
                book: str = "fund",
                against: str = "index") -> pd.DataFrame:
        """`expected` at every rebalance with a year of returns before it."""
        rows = {day: self.expected(day, book, against) for day in self.dates()}

        return pd.DataFrame.from_dict(rows, orient="index", columns=list(SUMMARY))

    def measured(self,
                 group: str | None = None) -> pd.DataFrame:
        """Each name's (or, with *group*, each group's) share of the fund's
        measured variance over the run, with ``other`` for the rest of its
        return, and its contribution to its measured volatility a year."""
        contributions = self._daily_contributions()

        if group is not None:
            contributions = self._grouped(contributions, group)

        fund = self.run.returns.reindex(contributions.index).astype(float)
        variance = float(fund.var())
        shares = contributions.apply(lambda column: column.cov(fund)) / variance
        volatility = float(np.sqrt(variance * self.run.periods_per_year))
        table = pd.DataFrame({"share": shares, "volatility": shares * volatility})

        return table.sort_values("share", ascending=False)

    def _model(self,
               day: pd.Timestamp) -> FactorRiskModel:
        model = self.models.at(day)
        assert isinstance(model, FactorRiskModel)

        return model

    def _vector(self,
                model: FactorRiskModel,
                day: pd.Timestamp,
                book: str,
                against: str | None) -> pd.Series:
        """The book's weights (less *against*'s) on the model's names."""
        weights = self.weights(day, book).reindex(model.asset_ids).fillna(0.0)

        if against is not None:
            weights = weights - self.weights(day, against).reindex(model.asset_ids).fillna(0.0)

        return weights

    def _groups(self,
                day: pd.Timestamp,
                field: str) -> pd.Series:
        """Each name's group on *day*, from the run's reference snapshots."""
        reference = self.run.tables[REFERENCE]
        dates = pd.DatetimeIndex(sorted(reference.index.get_level_values(0).unique()))
        at = as_of_position(dates, day)

        if at is None or field not in reference.columns:
            return pd.Series(dtype=object)

        rows = reference.loc[dates[at]]

        return rows.set_index("IDENTIFIER")[field].astype(str)

    def _daily_contributions(self) -> pd.DataFrame:
        """Each name's daily contribution to the fund's return, and the rest."""
        weights = self.run.tables[FUND_WEIGHTS].sort_index().fillna(0.0)
        weights.columns = weights.columns.astype(str)
        returns = self.run.tables[NAME_RETURNS].sort_index()
        returns.columns = returns.columns.astype(str)
        # Only the days the fund has a return after its first, each with the
        # weights at the close before; every column then covers the same days,
        # which is what makes the shares add up.
        fund = self.run.returns.iloc[1:].astype(float).dropna()
        days = fund.index.intersection(returns.index)
        before = weights.shift(1).reindex(days, method="ffill").fillna(0.0)
        names = [name for name in before.columns if name in returns.columns]
        found = (before[names] * returns.loc[days, names]).fillna(0.0)
        found = found.loc[:, (found != 0.0).any()]
        found[OTHER] = fund.loc[days] - found.sum(axis=1)

        return found

    def _grouped(self,
                 contributions: pd.DataFrame,
                 field: str) -> pd.DataFrame:
        """Daily contributions added up by each name's group as of each day."""
        names = [name for name in contributions.columns if name != OTHER]
        rebalances = pd.DatetimeIndex(self.models.dates)
        pieces = []

        for at, start in enumerate(rebalances):
            end = rebalances[at + 1] if at + 1 < len(rebalances) else None
            days = contributions.index[(contributions.index > start)
                                       & ((contributions.index <= end) if end is not None
                                          else True)]
            groups = self._groups(start, field).reindex(names).fillna("Unknown")
            pieces.append(contributions.loc[days, names].T.groupby(groups).sum().T)

        grouped = pd.concat(pieces).fillna(0.0)
        grouped[OTHER] = contributions[OTHER].reindex(grouped.index)

        return grouped


def _covariance_times(model: FactorRiskModel,
                      weights: pd.Series) -> pd.Series:
    """The model's covariance times *weights*, without forming it:
    ``B (F (B' w)) + D w``."""
    loadings = model.exposures.loc[weights.index].to_numpy(dtype=float)
    factor = model.factor_covariance.to_numpy(dtype=float)
    specific = model.specific_variance.reindex(weights.index).to_numpy(dtype=float)
    vector = weights.to_numpy(dtype=float)
    moved = loadings @ (factor @ (loadings.T @ vector)) + specific * vector

    return pd.Series(moved, index=weights.index)


def _split(model: FactorRiskModel,
           weights: pd.Series,
           prefix: str) -> dict[str, float]:
    """Volatility (or, with prefix ``tracking_``, tracking error) and its
    factor and specific parts."""
    loadings = model.exposures.loc[weights.index].to_numpy(dtype=float)
    vector = weights.to_numpy(dtype=float)
    exposure = loadings.T @ vector
    factor = float(exposure @ model.factor_covariance.to_numpy(dtype=float) @ exposure)
    specific = float((model.specific_variance.reindex(weights.index).to_numpy(dtype=float)
                      * vector * vector).sum())
    total = factor + specific
    name = "tracking_error" if prefix else "volatility"
    found = {name: float(np.sqrt(max(total, 0.0))),
             f"factor_{name}": float(np.sqrt(max(factor, 0.0))),
             f"specific_{name}": float(np.sqrt(max(specific, 0.0)))}

    if not prefix:
        found["factor_share"] = factor / total if total > 0.0 else 0.0

    return found


def _diversification(model: FactorRiskModel,
                     weights: pd.Series) -> dict[str, float]:
    """The diversification ratio and the implied average correlation."""
    loadings = model.exposures.loc[weights.index].to_numpy(dtype=float)
    factor = model.factor_covariance.to_numpy(dtype=float)
    own = (np.einsum("ij,jk,ik->i", loadings, factor, loadings)
           + model.specific_variance.reindex(weights.index).to_numpy(dtype=float))
    spread = np.sqrt(np.clip(own, 0.0, None))
    vector = weights.to_numpy(dtype=float)
    total = float(vector @ _covariance_times(model, weights).to_numpy())
    weighted = float(vector @ spread)
    alone = float(((vector * spread) ** 2).sum())
    sigma = float(np.sqrt(max(total, 0.0)))

    return {"diversification_ratio": weighted / sigma if sigma > 0.0 else 0.0,
            "average_correlation": ((total - alone) / (weighted ** 2 - alone)
                                    if weighted ** 2 - alone > 0.0 else 0.0)}
