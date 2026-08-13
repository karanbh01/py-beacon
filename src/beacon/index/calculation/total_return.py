# src/beacon/index/calculation/total_return.py
"""
Total-return and net-total-return accumulation.

`IndexCalculator` computed a price index and nothing else: on an ex-dividend
date a constituent's price drops, the aggregate drops with it, and the level
falls by the whole distribution. That is what a price index is *supposed* to
do, and it is why a price index understates what an investor actually earned.

BN-98 gave the data layer an action history and BN-118 gave every action a
`kind`. This is what consumes them.

## The construction: a divisor adjustment, not a purchase

The tempting implementation is to buy more units of the paying constituent with
its own dividend. That is wrong twice over: it silently re-weights the index
towards whatever paid, and it makes the composition depend on the return type,
so a price and a total-return version of one index would hold different things.

Index providers do it with the divisor instead. On an ex-date, with aggregate
holdings value ``A`` and total cash received ``D``:

    divisor_new = divisor_old x A / (A + D)

The level that day is then ``A / divisor_new == (A + D) / divisor_old`` — the
drop is exactly offset — and because the divisor is permanently smaller, every
later level is scaled up in the same proportion. That is reinvestment across
the index, and it leaves the units untouched.

## Only cash actions, and only once

A split is a ratio action: it changes the share count and the price together
and distributes nothing. Reinvesting a split ratio as though it were cash would
inflate the index by a factor of two. A rights issue, spin-off or merger is
structural and carries no directly aggregable value at all. So the filter is
`kind == "cash"`, using the classification the engine already publishes rather
than a list of type strings kept here.

The price path must already contain the drop for this to be right. It does: a
real feed quotes prices unadjusted, and BN-114's generator applies the ex-date
drop when it builds the close. Adding the cash back to an aggregate that never
fell would double-count the distribution.

## Net return

The same, with a flat withholding rate applied to the cash: ``D x (1 - rate)``.
A flat index-level rate rather than a per-country table, because a table keyed
off reference data is only as good as the country field behind it, and an
unpopulated table produces a number that looks precise and is not. The rate is
a property of the index and stated on it.
"""
import logging

import pandas as pd

from ...asset.base import Asset
from ...data.corporate_actions import CASH, kind_of
from ...data.fetcher import DataFetcher

logger = logging.getLogger(__name__)

# How returns accumulate.
PRICE = "PRICE"
TOTAL_RETURN = "TOTAL_RETURN"
NET_TOTAL_RETURN = "NET_TOTAL_RETURN"
RETURN_TYPES = (PRICE, TOTAL_RETURN, NET_TOTAL_RETURN)

# The types that accumulate distributions at all.
REINVESTING = (TOTAL_RETURN, NET_TOTAL_RETURN)


def withholding_for(return_type: str,
                    rate: float) -> float:
    """The fraction of each distribution withheld.

    Zero for a gross total-return index however the rate is set, so a
    definition carrying a rate it does not use cannot quietly apply it.
    """
    return rate if return_type == NET_TOTAL_RETURN else 0.0


class TotalReturnMixin:
    """Dividend reinvestment, mixed into IndexCalculator."""

    # Provided by the IndexCalculator that mixes this in.
    data: DataFetcher

    def rate_on(self,
                from_currency: str,
                to_currency: str,
                date: pd.Timestamp) -> float | None:
        """Provided by MarketValuesMixin; declared so this one type-checks."""
        raise NotImplementedError

    def cash_distribution_schedule(self) -> dict[pd.Timestamp, dict[str, float]]:
        """Every cash distribution the data source holds, by date and name.

        Built once per run rather than queried per constituent per day: a
        five-hundred-name index over five years would otherwise make more than
        half a million lookups to find a few thousand dividends.

        Returns:
            dict: ex-date -> {identifier: cash per share}. Empty when the data
            source holds no action history.
        """
        actions = self.data.corporate_actions
        if actions.is_empty:
            return {}

        frame = actions.data.reset_index(drop=True)
        cash = frame[frame["TYPE"].map(lambda value: kind_of(value) == CASH)]

        if cash.empty:
            logger.info("The action history holds no cash distributions.")

            return {}

        schedule: dict[pd.Timestamp, dict[str, float]] = {}
        for identifier, ex_date, value in zip(cash["IDENTIFIER"],
                                              cash["EX_DATE"],
                                              cash["VALUE"], strict=True):
            date = pd.Timestamp(ex_date)
            # Summed rather than assigned: a name can pay an ordinary and a
            # special dividend on one ex-date, and the second would otherwise
            # replace the first.
            per_date = schedule.setdefault(date, {})
            per_date[str(identifier)] = per_date.get(str(identifier), 0.0) + float(value)

        logger.info("Loaded cash distributions on %d ex-date(s).", len(schedule))

        return schedule

    @staticmethod
    def distribution_received(units: dict[Asset, float],
                              per_share: dict[str, float],
                              withholding: float = 0.0,
                              rates: dict[str, float] | None = None) -> float:
        """Cash the index's holdings receive on one date, in index currency.

        Args:
            units: What the index holds, asset to unit count.
            per_share: Cash per share by identifier, for this date, quoted in
                the paying company's own currency.
            withholding: Fraction withheld; 0.0 for a gross index.
            rates: Identifier to FX rate into the index currency. A missing
                entry converts at 1.0, which is correct for a name that
                already reports in the index currency.

        Returns:
            float: Total cash, net of withholding. Zero when nothing paid.
        """
        if not per_share:
            return 0.0

        rates = rates or {}

        gross = sum(count * per_share.get(asset.asset_id, 0.0)
                    * rates.get(asset.asset_id, 1.0)
                    for asset, count in units.items())

        return float(gross * (1.0 - withholding))

    def distribution_rates(self,
                           per_share: dict[str, float],
                           units: dict[Asset, float],
                           date: pd.Timestamp,
                           index_currency: str) -> dict[str, float]:
        """FX rates into the index currency, for the names paying on a date.

        Dividends are quoted in the paying company's currency while the
        aggregate they are reinvested into is in the index's, so the two
        cannot be divided until one is converted. Against a single-currency
        universe the omission is invisible -- every rate is 1.0 -- which is
        how it survived until the generator grew regions: a yen dividend was
        being counted as though it were dollars, and a twelve-name index came
        out with a 37% annual yield.

        Only the names that actually paid are looked up, so a quiet day costs
        nothing.

        The index currency is a parameter rather than read off
        ``self.definition``: `constructor` imports this module for its return
        types, so a mixin that reached back for the definition would close an
        import cycle. The caller has it already.
        """
        index_currency = index_currency.upper()
        held = {asset.asset_id: asset for asset in units}

        rates: dict[str, float] = {}
        date_str = date.strftime("%Y-%m-%d")

        for identifier in per_share:
            asset = held.get(identifier)

            if asset is None or asset.currency.upper() == index_currency:
                continue

            rate = self.rate_on(asset.currency, index_currency, date)

            if rate is None:
                logger.warning(
                    "No %s/%s rate on %s; treating the distribution from %s "
                    "as already in index currency.",
                    asset.currency, index_currency, date_str, identifier)
                continue

            rates[identifier] = rate

        return rates

    @staticmethod
    def reinvest(divisor: float,
                 aggregate: float,
                 distribution: float) -> float:
        """Shrink the divisor so a distribution is reinvested across the index.

        Args:
            divisor: The divisor in force.
            aggregate: Holdings value on the ex-date, after the price drop.
            distribution: Cash received, net of any withholding.

        Returns:
            float: The new divisor. Unchanged when nothing was distributed, or
            when the aggregate is non-positive — an index with no value cannot
            reinvest into itself, and scaling by zero would destroy the divisor
            rather than adjust it.
        """
        if distribution <= 0.0 or aggregate <= 0.0 or divisor <= 0.0:
            return divisor

        adjusted = divisor * aggregate / (aggregate + distribution)

        logger.debug("Reinvested %.4f into an aggregate of %.2f: divisor %.6f -> %.6f.",
                     distribution, aggregate, divisor, adjusted)

        return adjusted
