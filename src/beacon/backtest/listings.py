# src/beacon/backtest/listings.py
"""
What happens to a holding when its share count changes or its listing ends.

A split, reverse split or stock dividend multiplies the shares held by its
ratio on the ex-date, so the holding's value does not jump. A holding past
its last listed date is settled into cash at its last price, without cost.
Mixed into `BacktestEngine`.
"""
# Moved out of engine.py when it was split (BN-268); the behaviour is
# BN-244 (splits) and BN-197 (delistings).
import logging

import pandas as pd

from ..data.fetcher import DataFetcher
from ..portfolio.base import Portfolio

logger = logging.getLogger(__name__)


class ListingsMixin:
    """Share-count changes and delistings, mixed into BacktestEngine."""

    # Provided by the BacktestEngine that mixes this in.
    data_provider: DataFetcher

    def _delisting_dates(self) -> dict[str, pd.Timestamp]:
        """When each holding stops being listed, or an empty mapping.

        Defensive about what the provider *offers*, not about whether it
        works. The provider is an interface rather than a class, so a fetcher
        assembled by hand or stood in for by a double need not implement this,
        and a backtest over a universe where nothing is ever delisted should
        not require it to — hence the `getattr`, and hence a non-mapping answer
        being read as "nothing leaves".

        A failure is a different thing, and used to be swallowed into the same
        empty mapping with a WARNING (BN-197). Empty does not mean "unknown"
        here, it means "nothing is ever delisted", and the engine acts on it:
        the price read stops declining to carry a dead name forward, and
        disposal never settles the holding. So the book ran to the end holding
        names that no longer existed, each marked at its last close, and
        published a NAV as though that were real — with the only record of it
        in a log nobody reads after the fact.

        `IndexCalculator.delisting_schedule` calls the same method bare and
        always has. Two surfaces over one call answering differently is the
        BN-174 shape, and the one that substitutes is the one that was wrong.
        """
        getter = getattr(self.data_provider, "delisting_dates", None)

        if getter is None:
            return {}

        dates = getter()

        return dates if isinstance(dates, dict) else {}

    def _ratio_schedule(self) -> dict[pd.Timestamp, dict[str, float]]:
        """Every split and stock dividend in the data, by ex-date and name.

        Empty for a data provider with no action history, including one
        assembled by hand without a `corporate_actions` attribute.
        """
        actions = getattr(self.data_provider, "corporate_actions", None)

        if actions is None or actions.is_empty:
            return {}

        schedule: dict[pd.Timestamp, dict[str, float]] = actions.ratio_schedule()

        return schedule

    def _dispose_delisted(self,
                          portfolio: Portfolio,
                          date: pd.Timestamp,
                          delistings: dict[str, pd.Timestamp]) -> None:
        """Settle any holding whose listing has ended, into cash.

        Without this the position is held forever. A name past its last listed
        date has no price, so `_update_portfolio_prices` leaves the holding
        marked at its last close and `_sell_instruction` returns None rather
        than a trade -- the NAV keeps carrying a company that no longer
        exists, and its weight is never released to anything that does.

        The signal is *this mapping*, read from reference data's `DATE_TO`,
        and it always was: nothing here infers a delisting from prices running
        out, which is why BN-183 could change the price read without touching
        disposal. `_fetch_price` consults the same mapping and declines to
        carry a price forward past it, so the two agree on when a name stopped
        existing rather than one of them guessing from an absence.

        Settled at the last price the portfolio saw, and **without** a
        transaction cost. That is the modelling decision, and it is
        deliberate: an acquisition pays cash to the holder and a failure pays
        nothing, but neither is a trade crossed in a market that is by then
        closed. Charging brokerage on it would invent a fee nobody was
        billed.

        Args:
            portfolio: Mutated in place.
            date: Today.
            delistings: identifier -> last listed date.
        """
        if not delistings:
            return

        for asset_id in list(portfolio.holdings):
            last_listed = delistings.get(asset_id)

            if last_listed is None or date <= last_listed:
                continue

            holding = portfolio.holdings[asset_id]
            price = holding.current_price

            if price is None or price <= 0 or holding.quantity <= 0:
                logger.warning(
                    "[%s] %s delisted with no usable last price; the holding "
                    "is dropped and its value written off.", date, asset_id)
                portfolio.holdings.pop(asset_id, None)
                continue

            portfolio.execute_sell(asset_id, holding.quantity, price,
                                   cost=0.0, date=date)

            logger.info("[%s] Settled %.4f of %s at %.4f after delisting.",
                        date, holding.quantity, asset_id, price)
