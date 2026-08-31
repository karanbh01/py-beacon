# src/beacon/backtest/asset_view.py
"""
BacktestAssetView — one asset's story inside a run, read from the books.

Rebuilt for BN-154: the view reads the Portfolio (positions, transactions)
and the tracked index's book directly, instead of being handed flattened
frames. Two warts died in the move:

* `actual_weight_series()` is gone — it was a one-line alias of
  `weight_series()`, two names for one series.
* `holding_periods()` reads the positions panel (quantity > 0) rather than
  inferring holding from weight > 0 — a proxy that misread a position too
  small to round to a visible weight as "not held".
"""

import pandas as pd

from ..asset.view import AssetView
from ..data.fetcher import DataFetcher
from ..portfolio.base import Portfolio

# Imported lazily-typed to avoid a cycle: result.py imports this module.
# The view only needs the book's shape (levels/weights/source), so the
# annotation is deferred with a string.


class BacktestAssetView(AssetView):
    """AssetView with backtest context for a specific asset.

    Args:
        asset_id: The identifier used to look up data in the DataFetcher.
        data_fetcher: The data provider instance.
        portfolio: The run's books — positions, weights, transactions.
        index_book: The tracked index's book, when the run tracked one.
            Target weights come from its `source` snapshots: what each
            rebalance *decided*, which is the comparison slippage is about.
    """

    def __init__(self,
                 asset_id: str,
                 data_fetcher: DataFetcher,
                 portfolio: Portfolio,
                 index_book: "object | None" = None):
        super().__init__(asset_id, data_fetcher)
        self._portfolio = portfolio
        self._index_book = index_book

    # -- what the run did with this asset ---------------------------------

    def trades(self) -> pd.DataFrame:
        """This asset's transactions.

        Returns:
            pd.DataFrame: DataFrame with columns: date, type, quantity, price,
            cost. Empty DataFrame if no trades exist for this asset.
        """
        asset_txns = [t for t in self._portfolio.transactions
                      if t.asset_id == self._asset_id]

        if not asset_txns:
            return pd.DataFrame(columns=["date", "type", "quantity", "price",
                                         "cost"])

        rows = [
            {
                "date": t.transaction_date,
                "type": t.transaction_type,
                "quantity": t.quantity,
                "price": t.price,
                "cost": t.transaction_cost,
            }
            for t in asset_txns
        ]
        return pd.DataFrame(rows)

    def total_cost(self) -> float:
        """Sum of all transaction costs for this asset.

        Returns:
            float: Total transaction costs incurred for this asset.
        """
        return sum(t.transaction_cost for t in self._portfolio.transactions
                   if t.asset_id == self._asset_id)

    def holding_periods(self) -> list[dict[str, pd.Timestamp]]:
        """Continuous periods when this asset was held.

        Read from the positions panel — the record of quantities — rather
        than inferred from weights. Quantity above zero is the fact of
        holding; a weight of 0.0000 is a rounding statement about size.

        Returns:
            list of dict: Each dict has ``"start"`` and ``"end"`` keys with
            Timestamps. An open position at the end of the run has ``"end"``
            set to the last recorded date.
        """
        quantities = self._panel()["QUANTITY"]

        if quantities.empty:
            return []

        # Reindexed over the run's recorded calendar: the panel only carries
        # rows for dates the asset was held, so without the calendar a gap --
        # sold out, later re-bought -- would be invisible and two periods
        # would read as one.
        calendar = self._calendar()
        if not calendar.empty:
            quantities = quantities.reindex(calendar).fillna(0.0)

        held = quantities > 0
        periods = []
        in_period = False
        start = None
        prev_date = None

        for date, is_held in held.items():
            if is_held and not in_period:
                start = date
                in_period = True
            elif not is_held and in_period:
                periods.append({"start": start, "end": prev_date})
                in_period = False
            prev_date = date

        if in_period:
            periods.append({"start": start, "end": prev_date})

        return periods

    # -- weights: held versus decided -------------------------------------

    def weight_series(self) -> pd.Series:
        """Time series of this asset's portfolio weight.

        Returns:
            pd.Series: Weight at each date where the asset was held. Dates
            where the asset had zero or no weight are excluded.
        """
        weights = self._panel()["WEIGHT"]

        if weights.empty:
            return pd.Series(dtype=float)

        series = weights.dropna().astype(float)
        return series[series > 0]

    def target_weight_series(self) -> pd.Series:
        """Time series of this asset's target index weight.

        Read from the rebalance snapshots — what each rebalance decided —
        rather than the index's daily panel: the target the portfolio traded
        to is the snapshot, and the daily drift between rebalances is the
        index's business, not the portfolio's instruction.

        Returns:
            pd.Series: Target weight at each rebalance date. Rebalance dates
            where the asset was not a constituent are excluded. Empty
            Series if the run tracked no index.
        """
        snapshots = self._target_snapshots()

        if not snapshots:
            return pd.Series(dtype=float)

        data = {}
        for rebal_date in sorted(snapshots):
            weights = snapshots[rebal_date]
            if self._asset_id in weights:
                data[rebal_date] = weights[self._asset_id]
        return pd.Series(data, dtype=float)

    def slippage_vs_target(self) -> pd.Series:
        """Difference between actual and target weights over time.

        For each date the asset was held, finds the applicable target weight
        (most recent rebalance on or before that date) and computes
        actual - target.

        Returns:
            pd.Series: Slippage series indexed by date. Positive values mean
            the asset is overweight vs target. Empty Series if the run
            tracked no index.
        """
        snapshots = self._target_snapshots()

        if not snapshots:
            return pd.Series(dtype=float)

        actual = self.weight_series()
        if actual.empty:
            return pd.Series(dtype=float)

        sorted_rebal_dates = sorted(snapshots.keys())

        def _target_on_date(date: pd.Timestamp) -> float:
            applicable = [d for d in sorted_rebal_dates if d <= date]
            if not applicable:
                return 0.0
            latest = applicable[-1]
            return snapshots[latest].get(self._asset_id, 0.0)

        target = actual.index.to_series().apply(_target_on_date)
        target.index = actual.index
        return actual - target

    def weight_on_date(self,
                       date: pd.Timestamp) -> float | None:
        """This asset's portfolio weight on a specific date.

        Args:
            date: The query date.

        Returns:
            float or None: The weight, or None if the asset was not held on
            that date.
        """
        weights = self._panel()["WEIGHT"]

        if weights.empty:
            return None

        if date not in weights.index:
            # The books were written that day and this asset has no row: it
            # was not held, and falling back to an earlier weight would
            # report a position that had already been sold.
            calendar = self._calendar()
            if date in calendar:
                return None

            # Off the run calendar (a weekend, a holiday): the position in
            # force is the last recorded one, as it always was.
            applicable = weights.index[weights.index <= date]
            if applicable.empty:
                return None
            date = applicable[-1]

        value = weights.loc[date]
        if pd.isna(value) or value == 0:
            return None
        return float(value)

    # -- internals ---------------------------------------------------------

    def _panel(self) -> pd.DataFrame:
        """This asset's rows of the positions panel, indexed by date."""
        positions = self._portfolio.positions

        if positions.empty:
            return pd.DataFrame(columns=positions.columns)

        mine = positions[positions["ASSET_ID"] == self._asset_id]

        return mine.set_index("DATE").sort_index()

    def _calendar(self) -> pd.Index:
        """Every date the books were written, held or not."""
        return self._portfolio.nav.index

    def _target_snapshots(self) -> dict[pd.Timestamp, dict[str, float]]:
        """The tracked index's rebalance decisions, or nothing."""
        source = getattr(self._index_book, "source", None)

        if source is None:
            return {}

        snapshots: dict[pd.Timestamp, dict[str, float]] = source.weight_snapshots
        return snapshots

    def __repr__(self) -> str:
        return f"BacktestAssetView(asset_id='{self._asset_id}')"
