# tests/test_portfolio_batch.py
"""BN-344: a batch of same-day changes records the books once, with the same
history as recording after each change."""
import pandas as pd
import pytest

from beacon.portfolio.base import Portfolio

DAY = pd.Timestamp("2024-01-02")
NEXT = pd.Timestamp("2024-01-03")


def trade(portfolio: Portfolio) -> None:
    """A day's run of changes: buys, a sale, a price move and a dividend."""
    for name, quantity, price in (("AAA", 10, 50.0), ("BBB", 20, 25.0), ("CCC", 5, 100.0)):
        portfolio.execute_buy(name, quantity, price, cost=1.0, date=DAY)

    portfolio.execute_sell("BBB", 5, 26.0, cost=0.5, date=DAY)
    portfolio.update_prices({"AAA": 51.0, "BBB": 26.0, "CCC": 99.0}, DAY)
    portfolio.receive_cash(12.5, DAY, "DIVIDEND", "AAA")


def books(portfolio: Portfolio) -> tuple:
    return portfolio.positions, portfolio.cash, portfolio.nav


class TestABatch:

    def test_records_the_same_history(self):
        eager, batched = Portfolio("eager", 10_000.0), Portfolio("batched", 10_000.0)
        trade(eager)

        with batched.batch():
            trade(batched)

        for left, right in zip(books(eager), books(batched), strict=True):
            assert left.equals(right)

    def test_records_once_when_it_ends(self,
                                       monkeypatch):
        portfolio = Portfolio("p", 10_000.0)
        written = []
        record = portfolio._history.record
        monkeypatch.setattr(portfolio._history, "record",
                            lambda *args: written.append(args[0]) or record(*args))

        with portfolio.batch():
            trade(portfolio)
            assert written == []

        assert written == [DAY]

    def test_refuses_changes_on_two_dates(self):
        portfolio = Portfolio("p", 10_000.0)

        with pytest.raises(ValueError, match="one date"), portfolio.batch():
            portfolio.execute_buy("AAA", 1, 10.0, date=DAY)
            portfolio.execute_buy("AAA", 1, 10.0, date=NEXT)

    def test_inside_another_is_part_of_it(self,
                                          monkeypatch):
        portfolio = Portfolio("p", 10_000.0)
        written = []
        record = portfolio._history.record
        monkeypatch.setattr(portfolio._history, "record",
                            lambda *args: written.append(args[0]) or record(*args))

        with portfolio.batch():
            with portfolio.batch():
                portfolio.execute_buy("AAA", 1, 10.0, date=DAY)
            portfolio.execute_buy("BBB", 1, 10.0, date=DAY)

        assert written == [DAY]


class TestTheSnapshot:

    def test_keeps_its_columns_numeric(self):
        portfolio = Portfolio("p", 10_000.0)
        trade(portfolio)
        positions = portfolio.positions

        assert list(positions.columns) == ["DATE", "ASSET_ID", "QUANTITY", "PRICE",
                                           "MARKET_VALUE", "WEIGHT"]
        assert all(positions[column].dtype == "float64"
                   for column in ("QUANTITY", "PRICE", "MARKET_VALUE", "WEIGHT"))
        assert positions["WEIGHT"].sum() == pytest.approx(
            positions["MARKET_VALUE"].sum() / portfolio.nav.iloc[-1])


class TestTheWideWeights:
    """BN-361: the pivot is made once per change of the books."""

    def test_follow_the_books_as_they_change(self):
        portfolio = Portfolio("wide", 10_000.0)
        trade(portfolio)
        first = portfolio.weights

        portfolio.execute_buy("DDD", 4, 40.0, cost=0.0, date=NEXT)

        assert "DDD" not in first.columns
        assert portfolio.weights.loc[NEXT, "DDD"] > 0

    def test_are_made_once_while_the_books_stand_still(self,
                                                       monkeypatch):
        portfolio = Portfolio("still", 10_000.0)
        trade(portfolio)
        first = portfolio.weights
        pivots = []
        original = pd.DataFrame.pivot_table
        monkeypatch.setattr(pd.DataFrame, "pivot_table",
                            lambda *args, **kwargs: pivots.append(1) or original(*args, **kwargs))

        again = [portfolio.weights for _ in range(2)]

        assert pivots == []
        assert all(frame.equals(first) for frame in again)

    def test_a_caller_cannot_alter_the_kept_copy(self):
        portfolio = Portfolio("kept", 10_000.0)
        trade(portfolio)

        portfolio.weights.iloc[0, 0] = -1.0

        assert portfolio.weights.iloc[0, 0] >= 0
