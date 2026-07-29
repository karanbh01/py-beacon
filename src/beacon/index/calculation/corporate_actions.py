# src/beacon/index/calculation/corporate_actions.py
"""
Module for CorporateActionsMixin, responsible for adjusting the index
divisor in response to corporate actions.
"""
import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from ...asset.base import Asset
from ...asset.equity import Equity
from ...data.fetcher import DataFetcher
from ..constructor import IndexDefinition

logger = logging.getLogger(__name__)

class CorporateActionsMixin:
    """Corporate-action divisor adjustment logic, mixed into IndexCalculator."""

    # Corporate action types that are recognised but not yet implemented.
    _STUB_CA_TYPES = frozenset({"RIGHTS_ISSUE", "SPIN_OFF", "STOCK_DIVIDEND", "MERGER"})

    # Provided by the IndexCalculator that mixes this in.
    data: DataFetcher
    definition: IndexDefinition
    adjust_divisor_for_rebalance: Callable[[float, float, float], float]

    def handle_corporate_action(self,
                                action: dict[str, Any],
                                constituents: list[Asset],
                                current_total_market_value_before_ca: float,
                                current_divisor_before_ca: float) -> float:
        """Adjust the index divisor for a corporate action to maintain continuity.

        Currently supports **SPECIAL_DIVIDEND** fully.  Other action types
        (``RIGHTS_ISSUE``, ``SPIN_OFF``, ``STOCK_DIVIDEND``, ``MERGER``) are
        recognised stubs that log a warning and return the divisor unchanged.

        For a special dividend the market-value reduction is::

            reduction = dividend_per_share * shares_outstanding * ff * fx

        and the new divisor is::

            new_divisor = old_divisor * (mv_after / mv_before)

        where ``mv_after = mv_before - reduction``.

        Args:
            action: Dictionary with keys ``type``, ``asset``, ``value``,
                ``ex_date``.
            constituents: Current index constituents.
            current_total_market_value_before_ca: Aggregate market value of
                all constituents just before the action takes effect.
            current_divisor_before_ca: Divisor in effect before this action.

        Returns:
            The (possibly adjusted) divisor.
        """
        action_type = action.get('type', '').upper()
        asset_involved = action.get('asset')
        value = action.get('value')
        ex_date_raw = action.get('ex_date')

        if ex_date_raw is None:
            logger.warning(f"Corporate action missing ex_date: {action}. No divisor adjustment.")
            return current_divisor_before_ca
        ex_date = pd.Timestamp(ex_date_raw)

        logger.info(
            f"[{ex_date.strftime('%Y-%m-%d')}] Handling CA: {action_type} for asset "
            f"{asset_involved.asset_id if asset_involved else 'N/A'} "
            f"for index '{self.definition.index_name}'."
        )

        if asset_involved is None or value is None:
            logger.warning(
                f"Insufficient information for corporate action: {action}. "
                "No divisor adjustment.")
            return current_divisor_before_ca

        if asset_involved not in constituents:
            logger.info(
                f"Asset {asset_involved.asset_id} affected by CA is not currently "
                "an index constituent. No divisor adjustment."
            )
            return current_divisor_before_ca

        # --- Stub types: warn and return unchanged ---
        if action_type in self._STUB_CA_TYPES:
            logger.warning(
                f"Divisor adjustment for '{action_type}' is not yet implemented. "
                "Returning divisor unchanged."
            )
            return current_divisor_before_ca

        # --- SPECIAL_DIVIDEND ---
        if action_type == "SPECIAL_DIVIDEND":
            return self._special_dividend_divisor(
                asset_involved,
                value,
                ex_date,
                current_total_market_value_before_ca,
                current_divisor_before_ca,
            )

        # --- Unknown action type ---
        logger.warning(
            f"Unrecognised corporate action type '{action_type}'. "
            "Returning divisor unchanged."
        )
        return current_divisor_before_ca

    def _special_dividend_divisor(self,
                                  asset: Asset,
                                  value: float,
                                  ex_date: pd.Timestamp,
                                  mv_before: float,
                                  divisor_before: float) -> float:
        """Compute the divisor adjustment for a SPECIAL_DIVIDEND corporate action.

        Extracted from :meth:`handle_corporate_action` to keep nesting shallow;
        behaviour (including all log messages) is unchanged.
        """
        if not isinstance(asset, Equity):
            return divisor_before

        date_str = ex_date.strftime('%Y-%m-%d')

        shares = self.data.fetch_shares_outstanding(asset.ticker, date_str)
        if shares is None or shares <= 0:
            logger.warning(
                f"CA Handle: No shares for {asset.ticker}. "
                "Cannot adjust divisor for special dividend."
            )
            return divisor_before

        reduction_local = float(value) * shares

        # Apply free-float factor if the weighting scheme uses it
        if getattr(self.definition.weighting_scheme, 'use_free_float', False):
            ff = self.data.fetch_free_float_factor(asset.ticker, date_str)
            if ff is not None:
                reduction_local *= ff

        # FX conversion to index currency
        fx_rate = 1.0
        if asset.currency.upper() != self.definition.currency.upper():
            fx_series = self.data.fetch_fx_rates(
                asset.currency, self.definition.currency, date_str, date_str
            )
            if not fx_series.empty:
                fx_rate = float(fx_series.iloc[0])
            else:
                logger.warning(
                    f"CA Handle: No FX for {asset.currency}/"
                    f"{self.definition.currency}. Cannot adjust precisely."
                )
                return divisor_before

        reduction_index_ccy = reduction_local * fx_rate
        logger.debug(
            f"Special Dividend: Asset {asset.asset_id}, "
            f"reduction value (index ccy): {reduction_index_ccy:.2f}"
        )

        if abs(reduction_index_ccy) < 1e-9:
            logger.debug("Reduction is negligible. Divisor not changed.")
            return divisor_before

        if mv_before <= 0:
            logger.warning(
                f"CA Handle: Market value before CA is "
                f"{mv_before}. Cannot adjust divisor."
            )
            return divisor_before

        mv_after = mv_before - reduction_index_ccy
        if mv_after <= 0:
            logger.error(
                f"CA Handle: Market value after CA effect is non-positive "
                f"({mv_after}). Not adjusting divisor."
            )
            return divisor_before

        new_divisor = self.adjust_divisor_for_rebalance(
            divisor_before,
            mv_before,
            mv_after,
        )
        logger.info(
            f"Divisor adjusted due to SPECIAL_DIVIDEND for "
            f"{asset.asset_id}. Old: {divisor_before:.4f}, "
            f"New: {new_divisor:.4f}."
        )
        return new_divisor
