# src/beacon/index/calculation/deletions.py
"""
Removing a constituent that stopped being one part-way through a period.

An index reconstitutes on its rebalance dates, and between them it holds fixed
units. That works until a constituent is acquired, fails, or is otherwise
delisted — because from that day on there is no price, and the holding cannot
be valued at all.

## What goes wrong without this

`asset_unit_value` returns 0.0 when a price is missing, which is the right
answer to "what is this worth today" and the wrong basis for an index level. A
name that was 4% of the index simply stops contributing, so the level falls 4%
on the day it delists and never recovers it. The index reports a loss that no
holder experienced: in reality the position was sold, at a price, and the
proceeds stayed in the fund.

Nothing caught this before because no generated dataset had a delisting in it.
Every name listed on day one and never left, so the branch was unreachable.

## What a deletion actually is

The same divisor adjustment a rebalance uses. Value the book on the last day
the leaver had a price, once including it and once without:

    divisor ← divisor × (aggregate without) / (aggregate with)

The level is then identical across the change, which is the entire purpose of
a divisor. Holdings of the survivors are untouched, so their weights renormalise
upward in proportion — which is exactly what reinvesting the proceeds pro rata
across the remainder would have done.

## Why the reference data decides, not the price

A missing price and a delisting are different things. A gap in a feed is a
data-quality problem, and carrying the last level forward — which
`level_from_units` already does — is the right response. A name whose
reference record has *ended* is gone, and holding it forever is wrong.

Reading `DATE_TO` rather than guessing from absent prices is what keeps those
two apart, and it is why this consults the reference data at all rather than
just noticing the price vanish.
"""
import logging

import pandas as pd

from ...asset.base import Asset
from ...data.fetcher import DataFetcher

logger = logging.getLogger(__name__)


class DeletionMixin:
    """Intra-period constituent deletion, mixed into IndexCalculator."""

    # Provided by the IndexCalculator that mixes this in.
    data: DataFetcher

    def aggregate_value(self,
                        units: dict[Asset, float],
                        current_date: pd.Timestamp) -> float:
        """Provided by MarketValuesMixin; declared so this one type-checks."""
        raise NotImplementedError

    def delisting_schedule(self) -> dict[str, pd.Timestamp]:
        """The last date each identifier is listed, for those that end.

        Built once per run rather than queried per holding per day: a
        five-thousand-name index over ten years would otherwise make twelve
        million reference lookups to find a few hundred deletions.

        Returns:
            dict: identifier -> last listed date. Names still listed at the
            end of their record are absent, so an empty mapping means nothing
            ever leaves and the daily check costs one `if`.
        """
        schedule = self.data.delisting_dates()

        if schedule:
            logger.info("%d identifier(s) are delisted during their history.",
                        len(schedule))

        return schedule

    def apply_deletions(self,
                        units: dict[Asset, float],
                        divisor: float,
                        date: pd.Timestamp,
                        schedule: dict[str, pd.Timestamp],
                        valuation_date: pd.Timestamp
                        ) -> tuple[dict[Asset, float], float, list[str]]:
        """Drop any holding whose listing ended, keeping the level continuous.

        Args:
            units: What the index holds. Not mutated.
            divisor: The divisor in force.
            date: Today.
            schedule: Output of :meth:`delisting_schedule`.
            valuation_date: The last date the leavers still had prices —
                normally the previous trading day. Both aggregates are taken
                here, so the ratio is a like-for-like comparison rather than
                one that mixes today's prices with yesterday's.

        Returns:
            tuple: The surviving holdings, the adjusted divisor, and the
            identifiers removed. All three are unchanged when nothing left.
        """
        if not schedule:
            return units, divisor, []

        leaving = [asset for asset in units
                   if asset.asset_id in schedule
                   and date > schedule[asset.asset_id]]

        if not leaving:
            return units, divisor, []

        surviving = {asset: count for asset, count in units.items()
                     if asset not in leaving}

        removed = [asset.asset_id for asset in leaving]

        if not surviving:
            logger.error(
                "Every constituent was delisted by %s; keeping the holdings "
                "rather than emptying the index.", date.strftime("%Y-%m-%d"))

            return units, divisor, []

        before = self.aggregate_value(units, valuation_date)
        after = self.aggregate_value(surviving, valuation_date)

        if before <= 0.0 or after <= 0.0:
            logger.warning(
                "Cannot value the book on %s to delete %s; removing without "
                "a divisor adjustment, so the level will step.",
                valuation_date.strftime("%Y-%m-%d"), ", ".join(removed))

            return surviving, divisor, removed

        adjusted = divisor * after / before

        logger.info("Deleted %s on %s: divisor %.6f -> %.6f.",
                    ", ".join(removed), date.strftime("%Y-%m-%d"),
                    divisor, adjusted)

        return surviving, adjusted, removed
