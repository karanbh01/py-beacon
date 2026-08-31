# src/beacon/portfolio/asset_view.py
"""
PortfolioAssetView — one asset seen from a portfolio, position and market.

    p.asset("AAA").quantity            # from the books
    p.asset("AAA").unrealised_pnl
    p.asset("AAA").prices(start, end)  # from the data source
    p.asset("AAA").sector(on=date)     # point-in-time, not today's

The view reads **everything live** — position numbers from the portfolio
object, market facts from the data source — so nothing is cached that can go
stale. That is the same rule the redesign applied everywhere: a copied fact
can quietly disagree with its source, a read cannot.

The portfolio is typed as a Protocol rather than imported: `base.py` imports
this module to hand views out, so importing `Portfolio` back would be a
cycle — the same shape `history.py` avoided the same way (BN-152).
"""

from collections.abc import Mapping
from typing import Protocol

import pandas as pd

from ..asset.view import AssetView
from ..data.fetcher import DataFetcher


class _HoldingLike(Protocol):
    quantity: float
    average_cost_price: float
    current_price: float | None
    market_value: float | None


class _PortfolioLike(Protocol):
    # A read-only Mapping, not a dict: dict is invariant in its value type,
    # so `dict[str, Holding]` would not satisfy `dict[str, _HoldingLike]`
    # however compatible Holding is. The view only reads.
    @property
    def holdings(self) -> Mapping[str, _HoldingLike]: ...

    @property
    def positions(self) -> pd.DataFrame: ...

    def get_total_value(self) -> float: ...


class PortfolioAssetView(AssetView):
    """One asset's position and market data, read live.

    Args:
        asset_id: The identifier, as the portfolio and the data source
            both know it.
        data_fetcher: The data provider instance.
        portfolio: The books the position numbers come from.
    """

    def __init__(self,
                 asset_id: str,
                 data_fetcher: DataFetcher,
                 portfolio: _PortfolioLike):
        super().__init__(asset_id, data_fetcher)
        self._portfolio = portfolio

    # -- the position, live from the books ---------------------------------

    @property
    def quantity(self) -> float:
        """Units currently held; 0.0 for a position that was sold out."""
        holding = self._portfolio.holdings.get(self._asset_id)

        return float(holding.quantity) if holding is not None else 0.0

    @property
    def average_cost(self) -> float | None:
        """Weighted average cost of the current position, or None when the
        asset is not currently held."""
        holding = self._portfolio.holdings.get(self._asset_id)

        return (float(holding.average_cost_price) if holding is not None
                else None)

    @property
    def market_value(self) -> float | None:
        """The position's last marked value, or None when not held."""
        holding = self._portfolio.holdings.get(self._asset_id)

        if holding is None or holding.market_value is None:
            return None

        return float(holding.market_value)

    @property
    def weight(self) -> float:
        """The position's share of the portfolio right now; 0.0 when not
        held or unpriced."""
        value = self.market_value

        if value is None:
            return 0.0

        total = self._portfolio.get_total_value()

        return value / total if total != 0 else 0.0

    @property
    def unrealised_pnl(self) -> float | None:
        """What the current position has made against its average cost.

        None when the asset is not held or has never been priced — an
        unknown P&L is not a zero P&L.
        """
        holding = self._portfolio.holdings.get(self._asset_id)

        if holding is None or holding.current_price is None:
            return None

        return float((holding.current_price - holding.average_cost_price)
                     * holding.quantity)

    def position_history(self) -> pd.DataFrame:
        """This asset's rows of the positions panel, indexed by date."""
        positions = self._portfolio.positions

        if positions.empty:
            return pd.DataFrame(columns=positions.columns)

        return (positions[positions["ASSET_ID"] == self._asset_id]
                .set_index("DATE").sort_index())

    # -- the market, live from the source ----------------------------------

    def sector(self,
               on: pd.Timestamp | str | None = None) -> str | None:
        """The asset's sector as it stood on a date.

        Point-in-time through the fetcher's classification path: a name that
        moved sectors answers with the one in force on `on`, not today's.

        Args:
            on: The date to stand on; None reads the latest record.

        Returns:
            str or None: The sector, or None when reference data has none.
        """
        return self._data_fetcher.fetch_classification(self._asset_id, on)

    def __repr__(self) -> str:
        return (f"PortfolioAssetView(asset_id='{self._asset_id}', "
                f"quantity={self.quantity:g})")
