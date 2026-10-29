# src/beacon/analysis/risk/concentration.py
"""
How concentrated a book is: in its weights, in its risk and in what earned
its return.

**By weight**, over the names held (weight above 0), scaled to add to 1:

- the Herfindahl index ``H = sum w_i^2``, and its form normalised between
  an equally weighted book (0) and a single name (1),
  ``(H - 1/N) / (1 - 1/N)``;
- the effective number of holdings ``1 / H``;
- the largest weight and the top 5, 10 and 20 together;
- the fewest names making 50%, 80% and 90% of the book;
- optionally, the Gini coefficient from the Lorenz curve (0 equal, toward 1
  all in one name) and the entropy ``-sum w_i ln w_i``, with its effective
  number ``exp(E)``.

The same figures on group weights (a book's weight in each sector, say)
measure how concentrated it is across the groups.

**By risk**, from each name's share of the expected volatility at each
rebalance (`beacon.analysis.risk.contributions`): the Herfindahl of the
shares and the effective number of risk sources ``1 / sum p_i^2``, the top
names' share of the risk, the diversification ratio and the average
correlation; optionally the absorption ratio, the share of the variance in
the largest fifth of the covariance's eigenvalues.

**By outcome**, over a window: each name's contribution to the return (its
weight at the close before times its return, added up day by day), the top
contributors' share of the total, and breadth, the share of the names held
that added to it.

**Drift** is how far the weights have moved since the last rebalance: half
the sum of the gaps, the share of the book that would trade to put it back.

**The concentration curve** takes the names largest first, position by
position: the k-th name's weight, the cumulative weight of the top k (as a
share of the book), and the concentration of the top k on their own, their
Herfindahl and effective number once their weights are scaled to add to 1,
with how much adding the k-th changed each.
"""
# The concentration curve and top 25 followed on 29 October (Karan).
# BN-391. Agreed with Karan on 2026-10-27.
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ...data.base import as_of_position
from ..data import RunData
from ..panels import BENCHMARK_WEIGHTS, FUND_WEIGHTS, INDEX_WEIGHTS, NAME_RETURNS, REFERENCE
from ..performance.crises import THRESHOLD, CrisisAnalytics, Episode
from .contributions import RiskOverTime

BOOKS = {"fund": FUND_WEIGHTS, "index": INDEX_WEIGHTS, "benchmark": BENCHMARK_WEIGHTS}
TOPS = (5, 10, 20, 25)
COVERAGE = (0.5, 0.8, 0.9)

# The share of the eigenvalues the absorption ratio counts.
ABSORBED = 0.2


def weight_figures(weights: pd.DataFrame,
                   optional: bool = False) -> pd.DataFrame:
    """The concentration figures of each row of *weights* (a book's weights
    by date and name), a row each."""
    held = weights.fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    totals = held.sum(axis=1, keepdims=True)
    shares = np.divide(held, totals, out=np.zeros_like(held), where=totals > 0.0)
    counts = (shares > 0.0).sum(axis=1)
    ordered = -np.sort(-shares, axis=1)
    running = np.cumsum(ordered, axis=1)
    herfindahl = (shares ** 2).sum(axis=1)
    floor = 1.0 / np.maximum(counts, 1)
    normalised = np.where(counts > 1, (herfindahl - floor) / (1.0 - 1.0 / np.maximum(counts, 2)),
                          1.0)
    table = {"names": counts, "herfindahl": herfindahl, "normalised_herfindahl": normalised,
             "effective_number": np.divide(1.0, herfindahl, out=np.zeros_like(herfindahl),
                                           where=herfindahl > 0.0),
             "largest": ordered[:, 0] if ordered.shape[1] else np.zeros(len(held))}

    width = ordered.shape[1]

    for top in TOPS:
        table[f"top_{top}"] = running[:, min(top, width) - 1] if width else 0.0

    for share in COVERAGE:
        # The first count whose running total reaches the share; a little
        # tolerance so 0.8 is reached by weights that add to it exactly.
        table[f"names_for_{round(share * 100)}"] = (running < share - 1e-12).sum(axis=1) + 1

    if optional:
        table["gini"] = np.array([_gini(row[row > 0.0]) for row in shares])
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.where(shares > 0.0, shares * np.log(shares), 0.0)
        table["entropy"] = -logs.sum(axis=1)
        table["entropy_effective_number"] = np.exp(table["entropy"])

    return pd.DataFrame(table, index=weights.index)


def lorenz(weights: pd.Series) -> pd.DataFrame:
    """The Lorenz curve of the names held: the share of the names, smallest
    first, against the share of the book they make together."""
    held = np.sort(weights.fillna(0.0)[weights.fillna(0.0) > 0.0].to_numpy(dtype=float))
    total = held.sum()
    names = np.arange(len(held) + 1) / max(len(held), 1)
    book = np.concatenate([[0.0], np.cumsum(held) / total]) if total > 0 else names * 0.0

    return pd.DataFrame({"share_of_names": names, "share_of_book": book})


def concentration_curve(weights: pd.Series) -> pd.DataFrame:
    """The names held, largest first, a row a position from 1: each name,
    its weight (a share of the names held), the cumulative weight to it, and
    the top k's Herfindahl and effective number on their own, with the
    change the k-th made to each."""
    held = weights.fillna(0.0)
    held = held[held > 0.0].sort_values(ascending=False, kind="stable")
    total = float(held.sum())
    columns = ["name", "weight", "cumulative", "top_k_herfindahl", "top_k_effective_number",
               "marginal_herfindahl", "marginal_effective_number"]

    if total <= 0.0:
        return pd.DataFrame(columns=columns, index=pd.RangeIndex(1, 1, name="position"))

    shares = held.to_numpy(dtype=float) / total
    cumulative = np.cumsum(shares)
    herfindahl = np.cumsum(shares ** 2) / cumulative ** 2
    effective = 1.0 / herfindahl
    table = pd.DataFrame({"name": held.index.astype(str), "weight": shares,
                          "cumulative": cumulative, "top_k_herfindahl": herfindahl,
                          "top_k_effective_number": effective,
                          "marginal_herfindahl": np.diff(herfindahl, prepend=0.0),
                          "marginal_effective_number": np.diff(effective, prepend=0.0)},
                         index=pd.RangeIndex(1, len(shares) + 1, name="position"))

    return table[columns]


def risk_figures(shares: pd.Series,
                 tops: Sequence[int] = TOPS) -> dict[str, float]:
    """The concentration of a book's risk, from each name's share of it."""
    values = shares.dropna().astype(float)
    herfindahl = float((values ** 2).sum())
    ordered = values.sort_values(ascending=False)
    found = {"risk_herfindahl": herfindahl,
             "effective_risk_sources": 1.0 / herfindahl if herfindahl > 0.0 else 0.0}

    for top in tops:
        found[f"top_{top}_risk"] = float(ordered.iloc[:top].sum())

    return found


def outcome_figures(contributions: pd.Series,
                    held: int,
                    top: int = 10) -> dict[str, float | None]:
    """The top contributors' share of the return and breadth, from each
    name's contribution to it over a window."""
    total = float(contributions.sum())
    ordered = contributions.sort_values(ascending=False)

    return {"return": total,
            f"top_{top}_share": float(ordered.iloc[:top].sum()) / total if total != 0.0 else None,
            "breadth": float((contributions > 0.0).sum()) / held if held else None}


class ConcentrationAnalytics:
    """A run's concentration by weight, risk and outcome.

    Args:
        run: The run's data.
        risk: The run's risk over time, for the figures by risk.
    """

    def __init__(self,
                 run: RunData,
                 risk: RiskOverTime | None = None):
        self.run = run
        self.risk = risk

    def weights(self,
                book: str = "fund",
                field: str | None = None) -> pd.DataFrame:
        """A book's daily weights, by name or, with *field*, by group."""
        table = self.run.tables.get(BOOKS[book])

        if table is None or table.empty:
            raise KeyError(f"The run kept no weights for its {book}.")

        table = table.sort_index().fillna(0.0)
        table.columns = table.columns.astype(str)

        return table if field is None else self._grouped(table, field)

    def rebalances(self) -> list[pd.Timestamp]:
        """The run's rebalances."""
        reference = self.run.tables[REFERENCE]

        return sorted(pd.Timestamp(day) for day in reference.index.get_level_values(0).unique())

    def by_weight(self,
                  book: str = "fund",
                  field: str | None = None,
                  daily: bool = False,
                  optional: bool = False) -> pd.DataFrame:
        """The book's concentration by weight at each rebalance's close, or
        each day; across groups with *field*."""
        weights = self.weights(book, field)

        if not daily:
            rows = [as_of_position(weights.index, day) for day in self.rebalances()]
            weights = weights.iloc[[row for row in rows if row is not None]]

        return weight_figures(weights, optional)

    def against(self,
                other: str = "index",
                field: str | None = None,
                daily: bool = False) -> pd.DataFrame:
        """The fund's figures by weight less *other*'s, on the days both have."""
        ours = self.by_weight("fund", field, daily)
        theirs = self.by_weight(other, field, daily)
        days = ours.index.intersection(theirs.index)

        return ours.loc[days] - theirs.loc[days]

    def curve(self,
              date: str | pd.Timestamp,
              book: str = "fund",
              field: str | None = None) -> pd.DataFrame:
        """The concentration curve of a book's weights (or, with *field*,
        its group weights) at the close of *date*: names largest first, with
        cumulative weight and the top k's concentration."""
        weights = self.weights(book, field)
        at = as_of_position(weights.index, pd.Timestamp(date))

        if at is None:
            raise KeyError(f"The run kept no {book} weights on or before {date}.")

        return concentration_curve(weights.iloc[at])

    def by_period(self,
                  frequency: str = "Y",
                  book: str = "fund",
                  field: str | None = None) -> pd.DataFrame:
        """Each month's (``M``) or year's (``Y``) mean daily figures by weight."""
        daily = self.by_weight(book, field, daily=True)
        periods = pd.DatetimeIndex(daily.index).to_period(frequency).astype(str)

        return daily.groupby(periods).mean().rename_axis("period")

    def in_crises(self,
                  book: str = "fund",
                  field: str | None = None,
                  market: str | pd.Series | None = None,
                  threshold: float = THRESHOLD,
                  named: Sequence[Episode] | None = None) -> pd.DataFrame:
        """The mean daily figures by weight inside each crisis and on the
        calm days outside all of them."""
        daily = self.by_weight(book, field, daily=True)
        crises = CrisisAnalytics(self.run, market=market, threshold=threshold,
                                 named=named).crises()
        calm = pd.Series(True, index=daily.index)
        rows = {}

        for crisis in crises:
            inside = (daily.index > crisis.start) & (daily.index <= crisis.end)
            calm &= ~inside
            rows[crisis.name] = daily[inside].mean()

        rows["calm"] = daily[calm.to_numpy()].mean()

        return pd.DataFrame.from_dict(rows, orient="index")

    def drift(self,
              book: str = "fund") -> pd.Series:
        """Each day's drift from the last rebalance's weights."""
        weights = self.weights(book)
        rebalances = pd.DatetimeIndex(self.rebalances())
        found = {}

        for day in weights.index:
            at = as_of_position(rebalances, day)

            if at is None:
                continue

            start = as_of_position(weights.index, rebalances[at])
            gap = (weights.loc[day] - weights.iloc[start]).abs().sum() if start is not None else 0.0
            found[day] = 0.5 * float(gap)

        return pd.Series(found, name="drift")

    def by_risk(self,
                book: str = "fund") -> pd.DataFrame:
        """The book's concentration by risk at each rebalance the risk model
        covers, with its diversification ratio and average correlation.

        Raises:
            ValueError: If no risk was given.
        """
        risk = self._risk()
        rows = {}

        for day in risk.dates():
            figures = risk_figures(risk.contributions(day, book)["share"])
            expected = risk.expected(day, book)
            figures["diversification_ratio"] = expected["diversification_ratio"]
            figures["average_correlation"] = expected["average_correlation"]
            rows[day] = figures

        return pd.DataFrame.from_dict(rows, orient="index")

    def absorption_ratio(self,
                         date: str | pd.Timestamp,
                         share: float = ABSORBED) -> float:
        """The share of the variance of the names held at the rebalance on
        *date* in the largest *share* of the covariance's eigenvalues."""
        model = self._risk().models.at(pd.Timestamp(date))
        values = np.linalg.eigvalsh(model.covariance().to_numpy(dtype=float))[::-1]
        count = max(1, math.ceil(share * len(values)))

        return float(values[:count].sum() / values.sum()) if values.sum() > 0 else 0.0

    def by_outcome(self,
                   start: str | pd.Timestamp | None = None,
                   end: str | pd.Timestamp | None = None,
                   top: int = 10) -> dict[str, float | None]:
        """The top contributors' share of the fund's return from its names
        over the window (the whole run by default), and breadth."""
        contributions = self._contributions(start, end)
        held = int((contributions != 0.0).sum())

        return outcome_figures(contributions, held, top)

    def _contributions(self,
                       start: str | pd.Timestamp | None,
                       end: str | pd.Timestamp | None) -> pd.Series:
        """Each name's contribution to the fund's return over the window."""
        weights = self.weights("fund")
        returns = self.run.tables[NAME_RETURNS].sort_index()
        returns.columns = returns.columns.astype(str)
        days = self.run.returns.iloc[1:].index.intersection(returns.index)
        days = days[(days >= pd.Timestamp(start)) if start is not None else slice(None)]
        days = days[(days <= pd.Timestamp(end)) if end is not None else slice(None)]
        before = weights.shift(1).reindex(days, method="ffill").fillna(0.0)
        names = [name for name in before.columns if name in returns.columns]

        return (before[names] * returns.loc[days, names]).fillna(0.0).sum()

    def _grouped(self,
                 weights: pd.DataFrame,
                 field: str) -> pd.DataFrame:
        """Daily weights added up by each name's group as of the day."""
        reference = self.run.tables[REFERENCE]

        if field not in reference.columns:
            raise KeyError(f"The run kept no {field} for its names.")

        rebalances = pd.DatetimeIndex(self.rebalances())
        pieces = []

        for at, day in enumerate(rebalances):
            end = rebalances[at + 1] if at + 1 < len(rebalances) else None
            rows = weights[(weights.index >= day) & ((weights.index < end) if end is not None
                                                     else True)]
            groups = (reference.loc[day].set_index("IDENTIFIER")[field].astype(str)
                      .reindex(rows.columns).fillna("Unknown"))
            pieces.append(rows.T.groupby(groups).sum().T)

        first = weights[weights.index < rebalances[0]] if len(rebalances) else weights

        return pd.concat([first.iloc[:0], *pieces]).fillna(0.0) if pieces else first

    def _risk(self) -> RiskOverTime:
        if self.risk is None:
            raise ValueError("Concentration by risk needs the run's RiskOverTime.")

        return self.risk


def _gini(shares: np.ndarray) -> float:
    """The Gini coefficient of positive shares adding to 1."""
    if len(shares) < 2:
        return 0.0

    ordered = np.sort(shares)
    cumulative = np.cumsum(ordered)
    count = len(ordered)

    return float(1.0 - (2.0 * cumulative.sum() - cumulative[-1]) / (count * cumulative[-1]))
