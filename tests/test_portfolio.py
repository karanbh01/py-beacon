# tests/test_portfolio.py
"""Unit tests for Portfolio.execute_buy() and Portfolio.execute_sell()."""
import pytest
import pandas as pd

from beacon.portfolio.base import Portfolio, Transaction, Holding


AAPL = "AAPL"
MSFT = "MSFT"


@pytest.fixture
def portfolio():
    return Portfolio("test_portfolio", initial_cash=10000.0)


class TestExecuteBuy:

    def test_basic_buy(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=150.0)
        assert AAPL in portfolio.holdings
        assert portfolio.holdings[AAPL].quantity == 10
        assert portfolio.holdings[AAPL].average_cost_price == 150.0
        assert portfolio.cash_balance == pytest.approx(8500.0)

    def test_buy_deducts_cost(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0, cost=50.0)
        # 10 * 100 + 50 = 1050
        assert portfolio.cash_balance == pytest.approx(10000.0 - 1050.0)

    def test_buy_records_transaction(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=5, price=200.0, date=pd.Timestamp("2025-01-15"))
        assert len(portfolio.transactions) == 1
        tx = portfolio.transactions[0]
        assert tx.asset_id == AAPL
        assert tx.quantity == 5
        assert tx.price == 200.0
        assert tx.transaction_type == "BUY"
        assert tx.transaction_date == pd.Timestamp("2025-01-15")

    def test_buy_updates_market_data(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=150.0)
        assert portfolio.holdings[AAPL].current_price == 150.0
        assert portfolio.holdings[AAPL].market_value == pytest.approx(1500.0)

    def test_buy_adds_to_existing_holding(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_buy(AAPL, quantity=10, price=200.0)
        assert portfolio.holdings[AAPL].quantity == 20
        # avg cost = (10*100 + 10*200) / 20 = 150
        assert portfolio.holdings[AAPL].average_cost_price == pytest.approx(150.0)

    def test_insufficient_cash_skips_buy(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=1000, price=100.0)  # needs 100k, only 10k
        assert AAPL not in portfolio.holdings
        assert portfolio.cash_balance == 10000.0
        assert len(portfolio.transactions) == 0

    def test_buy_zero_quantity_raises(self, portfolio):
        with pytest.raises(ValueError, match="quantity must be positive"):
            portfolio.execute_buy(AAPL, quantity=0, price=100.0)

    def test_buy_negative_quantity_raises(self, portfolio):
        with pytest.raises(ValueError, match="quantity must be positive"):
            portfolio.execute_buy(AAPL, quantity=-5, price=100.0)

    def test_buy_negative_price_raises(self, portfolio):
        with pytest.raises(ValueError, match="price cannot be negative"):
            portfolio.execute_buy(AAPL, quantity=5, price=-10.0)

    def test_buy_defaults_date_to_now(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=1, price=100.0)
        assert len(portfolio.transactions) == 1
        assert isinstance(portfolio.transactions[0].transaction_date, pd.Timestamp)


class TestExecuteSell:

    def test_basic_sell(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_sell(AAPL, quantity=5, price=120.0)
        assert portfolio.holdings[AAPL].quantity == 5
        # cash: 10000 - 1000 + 600 = 9600
        assert portfolio.cash_balance == pytest.approx(9600.0)

    def test_sell_deducts_cost(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_sell(AAPL, quantity=5, price=120.0, cost=10.0)
        # cash: 10000 - 1000 + (600 - 10) = 9590
        assert portfolio.cash_balance == pytest.approx(9590.0)

    def test_sell_records_transaction(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_sell(AAPL, quantity=5, price=120.0, date=pd.Timestamp("2025-02-01"))
        assert len(portfolio.transactions) == 2  # buy + sell
        tx = portfolio.transactions[1]
        assert tx.transaction_type == "SELL"
        assert tx.quantity == 5
        assert tx.price == 120.0

    def test_full_sell_removes_holding(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_sell(AAPL, quantity=10, price=120.0)
        assert AAPL not in portfolio.holdings

    def test_insufficient_holdings_skips_sell(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=5, price=100.0)
        initial_cash = portfolio.cash_balance
        portfolio.execute_sell(AAPL, quantity=10, price=120.0)  # only have 5
        assert portfolio.holdings[AAPL].quantity == 5  # unchanged
        assert portfolio.cash_balance == initial_cash  # unchanged
        assert len(portfolio.transactions) == 1  # only the buy

    def test_sell_asset_not_held_skips(self, portfolio):
        portfolio.execute_sell(MSFT, quantity=5, price=100.0)
        assert len(portfolio.transactions) == 0
        assert portfolio.cash_balance == 10000.0

    def test_sell_zero_quantity_raises(self, portfolio):
        with pytest.raises(ValueError, match="quantity must be positive"):
            portfolio.execute_sell(AAPL, quantity=0, price=100.0)

    def test_sell_negative_price_raises(self, portfolio):
        with pytest.raises(ValueError, match="price cannot be negative"):
            portfolio.execute_sell(AAPL, quantity=5, price=-10.0)


class TestPortfolioIntegration:

    def test_buy_sell_cycle(self, portfolio):
        """Buy, sell partial, verify state, sell rest."""
        portfolio.execute_buy(AAPL, quantity=20, price=50.0)
        assert portfolio.cash_balance == pytest.approx(9000.0)

        portfolio.execute_sell(AAPL, quantity=10, price=60.0)
        assert portfolio.cash_balance == pytest.approx(9600.0)
        assert portfolio.holdings[AAPL].quantity == 10

        portfolio.execute_sell(AAPL, quantity=10, price=70.0)
        assert portfolio.cash_balance == pytest.approx(10300.0)
        assert AAPL not in portfolio.holdings

    def test_multiple_assets(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_buy(MSFT, quantity=5, price=200.0)
        assert len(portfolio.holdings) == 2
        assert portfolio.cash_balance == pytest.approx(10000.0 - 1000.0 - 1000.0)

    def test_get_total_value_after_trades(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.update_prices({"AAPL": 110.0})
        # holdings: 10 * 110 = 1100, cash: 9000
        assert portfolio.get_total_value() == pytest.approx(10100.0)

    def test_transactions_list_complete(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_buy(MSFT, quantity=5, price=200.0)
        portfolio.execute_sell(AAPL, quantity=5, price=110.0)
        assert len(portfolio.transactions) == 3
        assert all(isinstance(tx, Transaction) for tx in portfolio.transactions)
