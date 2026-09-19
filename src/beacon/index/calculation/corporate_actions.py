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
from ...asset.equity import require_equity
from ...data.fetcher import DataFetcher
from ...exceptions import CalculationError
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

        Currently supports **SPECIAL_DIVIDEND** fully.  Other recognised types
        (``RIGHTS_ISSUE``, ``SPIN_OFF``, ``STOCK_DIVIDEND``, ``MERGER``) are
        unimplemented and are refused.

        Returning the divisor unchanged is this method's answer for *an action
        that genuinely has no effect on the index* — one affecting a name the
        index does not hold, or one whose adjustment rounds to nothing. Under
        BN-184 it is no longer also the answer for "this action was malformed",
        "this type is not implemented" or "this type is unknown": an
        unadjustable action and a harmless one must not be spelled the same,
        because the second is safe to publish and the first leaves the level
        wrong from that day onward.

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

        Raises:
            CalculationError: If the action is malformed, if its type is
                unknown or recognised-but-unimplemented, or if the affected
                asset is a constituent and is not an equity (BN-185).
        """
        action_type = action.get('type', '').upper()
        asset_involved = action.get('asset')
        value = action.get('value')
        ex_date_raw = action.get('ex_date')

        if ex_date_raw is None:
            raise CalculationError(
                calculation_name="CorporateActionDivisor",
                details=(f"the corporate action {action!r} carries no ex_date, "
                         f"so there is no day to apply it on. It used to be "
                         f"dropped with a warning, which is how a harmless "
                         f"action is spelled."))

        ex_date = pd.Timestamp(ex_date_raw)

        logger.info(
            f"[{ex_date.strftime('%Y-%m-%d')}] Handling CA: {action_type} for asset "
            f"{asset_involved.asset_id if asset_involved else 'N/A'} "
            f"for index '{self.definition.index_name}'."
        )

        if asset_involved is None or value is None:
            raise CalculationError(
                calculation_name="CorporateActionDivisor",
                details=(f"the corporate action {action!r} is missing its "
                         f"asset or its value, so the market-value reduction "
                         f"it implies cannot be computed. It used to be "
                         f"dropped with a warning, which is how a harmless "
                         f"action is spelled."))

        # The one genuinely-no-effect case, and the reason this method has a
        # "divisor unchanged" answer at all: an action on a name the index
        # does not hold moves none of the index's market value, so the
        # divisor that preserves continuity is the one already in force.
        # Logged at INFO, not WARNING, because nothing went wrong.
        if asset_involved not in constituents:
            logger.info(
                f"Asset {asset_involved.asset_id} affected by CA is not currently "
                "an index constituent. No divisor adjustment."
            )
            return current_divisor_before_ca

        # --- Recognised but unimplemented ---
        # These used to return the divisor unchanged, which is also what a
        # no-effect action returns — so "Beacon cannot adjust for this" and
        # "this action does not move the index" were indistinguishable, and a
        # spin-off silently left the level wrong forever after (BN-184).
        if action_type in self._STUB_CA_TYPES:
            raise CalculationError(
                calculation_name="CorporateActionDivisor",
                details=(f"divisor adjustment for '{action_type}' is not "
                         f"implemented, so the effect of this action on "
                         f"{asset_involved.asset_id} cannot be computed. "
                         f"Leaving the divisor unchanged would publish it as "
                         f"an action with no effect on the index."))

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
        raise CalculationError(
            calculation_name="CorporateActionDivisor",
            details=(f"corporate action type '{action_type}' on "
                     f"{asset_involved.asset_id} is not recognised, so its "
                     f"effect on the index is unknown. An unknown action is "
                     f"not a no-effect action, which is what returning the "
                     f"divisor unchanged would say."))

    def _special_dividend_divisor(self,
                                  asset: Asset,
                                  value: float,
                                  ex_date: pd.Timestamp,
                                  mv_before: float,
                                  divisor_before: float) -> float:
        """Compute the divisor adjustment for a SPECIAL_DIVIDEND corporate action.

        Extracted from :meth:`handle_corporate_action` to keep nesting shallow.

        Raises:
            CalculationError: If *asset* is not an equity (BN-185), or if the
                shares, FX rate or market value the adjustment needs are
                missing or unusable (BN-184). These all used to return the
                divisor unchanged, which is also what a no-op action returns —
                so an unadjustable action and a harmless one were spelled
                identically, and a divisor that should have moved silently did
                not.
        """
        equity = require_equity(asset, "CorporateActionDivisor",
                                "carry a divisor adjustment")

        date_str = ex_date.strftime('%Y-%m-%d')

        shares = self.data.fetch_shares_outstanding(equity.ticker, date_str)
        if shares is None or shares <= 0:
            raise CalculationError(
                calculation_name="CorporateActionDivisor",
                details=(f"no shares outstanding for {equity.ticker} on "
                         f"{date_str}, so the special dividend's market-value "
                         f"reduction cannot be sized. The dividend happened "
                         f"either way: skipping the adjustment leaves the "
                         f"level overstated from this day onward."))

        reduction_local = float(value) * shares

        # Apply free-float factor if the weighting scheme uses it
        if getattr(self.definition.weighting_scheme, 'use_free_float', False):
            ff = self.data.fetch_free_float_factor(equity.ticker, date_str)
            if ff is not None:
                reduction_local *= ff

        # FX conversion to index currency, through the library's one lookup
        # (BN-207). This used to fetch `date_str..date_str` and take the single
        # row, which is an exact-day read written inline -- correct in itself,
        # and the reason this site survived BN-188's consolidation. It also
        # meant a dividend on a day the FX feed happened to skip refused where
        # every other conversion carried forward, and nothing said which
        # behaviour was intended. That choice is now `fx_policy`, set once for
        # the dataset: EXACT_DAY reproduces exactly what this line did.
        fx_rate = self.data.fx_rate_on(asset.currency,
                                       self.definition.currency,
                                       ex_date)

        if fx_rate is None:
            raise CalculationError(
                calculation_name="CorporateActionDivisor",
                details=(f"no {asset.currency}/{self.definition.currency} "
                         f"rate on {date_str} under the {self.data.fx_policy} "
                         f"policy, so a dividend paid in {asset.currency} "
                         f"cannot be expressed in the index's money. BN-188 "
                         f"refused the same gap in the weighting; skipping "
                         f"the adjustment here leaves the level overstated "
                         f"instead."))

        reduction_index_ccy = reduction_local * fx_rate
        logger.debug(
            f"Special Dividend: Asset {asset.asset_id}, "
            f"reduction value (index ccy): {reduction_index_ccy:.2f}"
        )

        # Genuinely no effect: the continuity adjustment is
        # `divisor * (mv - reduction) / mv`, which for a reduction of zero is
        # the divisor itself. Returning it unchanged is the arithmetic, not a
        # substitute for it, so this one stays (BN-184).
        if abs(reduction_index_ccy) < 1e-9:
            logger.debug("Reduction is negligible. Divisor not changed.")
            return divisor_before

        if mv_before <= 0:
            raise CalculationError(
                calculation_name="CorporateActionDivisor",
                details=(f"the index is worth {mv_before} before this action, "
                         f"so the continuity ratio it needs is undefined. "
                         f"`adjust_divisor_for_rebalance` refuses the same "
                         f"input one call down; returning the divisor "
                         f"unchanged here quietly stepped around it."))

        mv_after = mv_before - reduction_index_ccy

        if mv_after <= 0:
            raise CalculationError(
                calculation_name="CorporateActionDivisor",
                details=(f"a reduction of {reduction_index_ccy} against a "
                         f"market value of {mv_before} leaves {mv_after}, so "
                         f"the action as given consumes the whole index. "
                         f"Either the dividend or the market value is wrong, "
                         f"and leaving the divisor alone publishes both."))

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
