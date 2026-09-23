# tests/conftest.py
"""Shared test support.

`wire_fetch_price` exists because of BN-212, and the reason is worth keeping.

The index calculator and the backtest engine both priced one name on one date
by calling `fetch_market_data` and taking `.iloc[0]` off the frame it returned.
That read never consults the session panel, so a run made one full-frame slice
per name per session -- 172,000 of them over a 200-name three-year run. Moving
both to `fetch_price`, which does consult it, took the pair from 158 seconds to
30.

Seven test files mock a `DataFetcher` with a `MagicMock` and configure
`fetch_market_data` on it. None configured `fetch_price`, so all of them broke
at once -- a Mock is returned, multiplied into a valuation, and surfaces
several frames later as a comparison between a Mock and a float.

Fifth time this fortnight that a partial double broke on code moving, and the
first time the count was high enough to be a signal rather than a nuisance:
seven fixtures each asserting that `fetch_market_data` is the whole of how a
price is read. This derives one from the other in one place, so the next method
to move has one fixture to update rather than seven.
"""
import pandas as pd


def wire_fetch_price(provider,
                     price_column: str = "CLOSE") -> None:
    """Make a mocked provider answer `fetch_price` from its `fetch_market_data`.

    Mirrors what the real `DataFetcher` does -- read the column on the exact
    date, None when absent or NaN -- so a fixture that has configured prices
    one way gets the other for free and the two cannot disagree.

    Args:
        provider: A `MagicMock` standing in for a `DataFetcher`.
        price_column: The column `fetch_price` defaults to reading.
    """
    def fetch_price(identifier,
                    date,
                    column=price_column):
        frame = provider.fetch_market_data(identifier, date, date)

        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return None

        if column not in frame.columns:
            return None

        value = frame[column].iloc[0]

        return None if pd.isna(value) else float(value)

    provider.fetch_price.side_effect = fetch_price

    # BN-218 moved the daily valuation to a batch read. Derived from
    # `fetch_price` rather than configured separately, so the two cannot
    # disagree -- and this is the payoff of keeping the derivation here: the
    # method moved, and one line changed instead of seven fixtures.
    def prices_on(identifiers,
                  date,
                  column=price_column):
        return {name: fetch_price(name, date, column) for name in identifiers}

    provider.prices_on.side_effect = prices_on
