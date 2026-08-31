# src/beacon/index/result.py
"""
IndexResult — output container for index calculation results.
"""
from dataclasses import dataclass, field

import pandas as pd

from ..data.fetcher import DataFetcher
from ..plot.base import PlotAccessor
from .asset_view import IndexAssetView
from .capping import CapReport

#: Long-form columns of the daily weights panel, in order.
DAILY_WEIGHT_COLUMNS = ["DATE", "IDENTIFIER", "AMOUNT", "WEIGHT"]

# Storage dtypes, chosen for the size of a real run rather than for a test
# panel. A ten-year index over 6,000 names is ~15.1M rows, so every byte a row
# costs ~15 MB. Measured on a 60,000-row panel (see tests/test_daily_weights.py,
# which asserts the figure rather than trusting this comment):
#
#   these dtypes    18.9 bytes/row   ->  ~286 MB at 6,000 names x 10 years
#   pandas defaults 78.0 bytes/row   ->  ~1.18 GB for the same panel
#
# The difference is almost all the identifier — 2 bytes as a categorical code
# against 54 as a string — plus 4 bytes a value rather than 8, which is
# ~60 MB a float32 column against ~121 MB.
#
# float32 carries about seven significant digits, which is far more than a
# weight needs and enough for an amount: the divisor identity it has to
# support reconstructs the index level to a relative error near 1e-6.
_DAILY_WEIGHT_DTYPES = {"DATE": "datetime64[ns]",
                        "IDENTIFIER": "category",
                        "AMOUNT": "float32",
                        "WEIGHT": "float32"}


def daily_weights_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    """Build the daily weights panel from records collected during a run.

    Args:
        records: One dict per constituent per calculation day, with the keys
            in :data:`DAILY_WEIGHT_COLUMNS`.

    Returns:
        pd.DataFrame: Long-form ``DATE``/``IDENTIFIER``/``AMOUNT``/``WEIGHT``,
        in the storage dtypes. Empty records give an empty frame that still
        carries the columns, so a consumer can slice it without checking.
    """
    return pd.DataFrame(records,
                        columns=DAILY_WEIGHT_COLUMNS).astype(_DAILY_WEIGHT_DTYPES)


def empty_daily_weights() -> pd.DataFrame:
    """The panel an :class:`IndexResult` carries when nothing recorded one."""
    return daily_weights_frame([])


@dataclass
class IndexResult:
    """Container holding the output of an index calculation run.

    Args:
        index_id: Identifier of the calculated index.
        index_levels: Time series of index levels indexed by
            ``pd.DatetimeIndex``.
        divisor_history: Time series of divisor values indexed by
            ``pd.DatetimeIndex``.
        constituent_snapshots: Mapping of rebalance date -> list of
            asset_id strings.
        weight_snapshots: Mapping of rebalance date -> dict of
            {asset_id: weight}.
        cap_reports: Mapping of rebalance date -> CapReport, for the
            rebalances where a weight cap actually bound. Empty for an
            uncapped index, so its presence is itself the signal that
            capping occurred.
        announcement_dates: Mapping of *effective* date -> the date that
            composition was announced. Snapshots are keyed by the effective
            date, because that is when the weights are in force and what every
            consumer — drift, attribution, the backtest engine — needs. The
            announcement is carried alongside rather than instead, since a
            client showing "rebalance of 18 Sep, effective 22 Sep" needs both.
            Empty for an index with no lag, where the two always coincide.
        daily_weights: Long-form panel of what the index held on every
            calculation day: ``DATE``, ``IDENTIFIER``, ``AMOUNT`` (units held)
            and ``WEIGHT`` (that holding's share of the day's aggregate
            value). Recorded by the calculator as it walks, not derived
            afterwards — see the note below. Defaults to an empty frame, so a
            result built by hand or by an older caller is still valid.

    The daily panel is *recorded* rather than re-derived because the index's
    daily state is path-dependent. It is not a forward-fill of the rebalance
    snapshot, and not even "amounts fixed between rebalances, repriced daily":
    :class:`~beacon.index.calculation.deletions.DeletionMixin` drops a delisted
    name mid-period and adjusts the divisor, and
    :class:`~beacon.index.calculation.corporate_actions.CorporateActionsMixin`
    adjusts it on ex-dates. Both change what is held and what each name weighs
    on a day that is not a rebalance. A path is written down as it happens.

    The rebalance snapshots stay what they always were: the record of what a
    rebalance *decided*. This panel is the record of what then *happened*.
    """

    #: Charts for this result. A descriptor that resolves on first
    #: access, so matplotlib is imported only when something is drawn.
    plot = PlotAccessor("IndexPlots")
    index_id: str
    index_levels: pd.Series
    divisor_history: pd.Series
    constituent_snapshots: dict[pd.Timestamp, list[str]]
    weight_snapshots: dict[pd.Timestamp, dict[str, float]]
    cap_reports: dict[pd.Timestamp, CapReport] = field(default_factory=dict)
    announcement_dates: dict[pd.Timestamp, pd.Timestamp] = field(
        default_factory=dict)
    # compare=False: a frame compares element-wise, so leaving it in the
    # generated __eq__ would make comparing two results raise rather than
    # answer.
    daily_weights: pd.DataFrame = field(default_factory=empty_daily_weights,
                                        repr=False, compare=False)
    _data_fetcher: DataFetcher | None = field(default=None, repr=False, compare=False)

    def capped_assets_on_date(self,
                              date: pd.Timestamp) -> dict[str, float]:
        """Return the constituents held at the cap at the given rebalance.

        Args:
            date: A rebalance date.

        Returns:
            dict: ``{asset_id: uncapped_weight}`` for names the cap bound on
            that date. Empty when nothing was capped, or when *date* is not a
            rebalance date.
        """
        report = self.cap_reports.get(date)

        return dict(report.capped) if report is not None else {}

    def with_data(self,
                  data_fetcher: DataFetcher) -> 'IndexResult':
        """Bind a DataFetcher for asset-level queries. Returns self for chaining."""
        self._data_fetcher = data_fetcher
        return self

    def asset(self,
              asset_id: str) -> IndexAssetView:
        """Return an IndexAssetView for a constituent.

        Args:
            asset_id: Identifier of the constituent asset.

        Returns:
            IndexAssetView

        Raises:
            RuntimeError: If no DataFetcher has been bound via
                :meth:`with_data`.
            KeyError: If *asset_id* is not found in any constituent
                snapshot.
        """
        if self._data_fetcher is None:
            raise RuntimeError(
                "No DataFetcher bound. Call .with_data(fetcher) first."
            )

        all_constituents = set()
        for ids in self.constituent_snapshots.values():
            all_constituents.update(ids)

        if asset_id not in all_constituents:
            raise KeyError(
                f"Asset '{asset_id}' not found in any constituent snapshot."
            )

        return IndexAssetView(
            asset_id=asset_id,
            data_fetcher=self._data_fetcher,
            weight_snapshots=self.weight_snapshots,
            index_levels=self.index_levels,
        )

    def get_returns(self) -> pd.Series:
        """Derive a return series from index levels.

        Returns:
            pd.Series: Percentage returns (first entry is dropped).
        """
        if self.index_levels.empty:
            return pd.Series(dtype=float)
        return self.index_levels.pct_change().dropna()

    def get_weights_on_date(self,
                            date: pd.Timestamp) -> dict[str, float]:
        """Get constituent weights effective on a given date.

        Locates the most recent rebalance date on or before *date*.

        Args:
            date: The query date.

        Returns:
            dict: Mapping of asset_id to weight. Empty dict if no rebalance
            has occurred on or before *date*.
        """
        applicable_dates = [d for d in self.weight_snapshots if d <= date]
        if not applicable_dates:
            return {}
        latest = max(applicable_dates)
        return self.weight_snapshots[latest]

    def weights_on(self,
                   date: pd.Timestamp) -> dict[str, float]:
        """Get the *recorded* constituent weights as of a given date.

        Reads the daily panel rather than the rebalance snapshots, so the
        answer includes everything that happened since the last rebalance:
        price drift, a deletion, a divisor adjustment. Compare
        :meth:`get_weights_on_date`, which answers the different question of
        what the last rebalance decided.

        Falls back to the latest recorded date on or before *date* — which
        covers a day the holdings could not be valued at all, since such a day
        records no rows.

        Args:
            date: The query date.

        Returns:
            dict: Mapping of identifier to weight. Empty when nothing was
            recorded on or before *date*, including when no panel was captured
            at all.
        """
        panel = self.daily_weights
        if panel.empty:
            return {}

        recorded = panel["DATE"] <= date
        if not recorded.any():
            return {}

        latest = panel.loc[recorded, "DATE"].max()
        rows = panel.loc[panel["DATE"] == latest]

        return {str(identifier): float(weight)
                for identifier, weight in zip(rows["IDENTIFIER"],
                                              rows["WEIGHT"],
                                              strict=True)}

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten index levels and divisor history into a DataFrame.

        Returns:
            pd.DataFrame: Columns: ``index_level``, ``divisor``.
        """
        df = pd.DataFrame({
            "index_level": self.index_levels,
            "divisor": self.divisor_history,
        })
        df.index.name = "date"
        return df

    def __repr__(self) -> str:
        n_dates = len(self.index_levels)
        n_rebalances = len(self.weight_snapshots)
        bound = self._data_fetcher is not None
        return (
            f"IndexResult(index_id='{self.index_id}', "
            f"dates={n_dates}, rebalances={n_rebalances}, "
            f"data_bound={bound})"
        )
