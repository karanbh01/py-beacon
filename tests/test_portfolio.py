# tests/test_portfolio.py
"""Unit tests for the refactored Portfolio class."""
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


class TestConstruction:

    def test_valid_construction(self):
        p = Portfolio("p1", initial_cash=5000.0)
        assert p.portfolio_id == "p1"
        assert p.cash_balance == 5000.0
        assert p.holdings == {}
        assert p.transactions == []

    def test_zero_initial_cash(self):
        p = Portfolio("p1", initial_cash=0.0)
        assert p.cash_balance == 0.0

    def test_default_initial_cash(self):
        p = Portfolio("p1")
        assert p.cash_balance == 0.0

    def test_negative_initial_cash_raises(self):
        with pytest.raises(ValueError, match="Initial cash cannot be negative"):
            Portfolio("p1", initial_cash=-100.0)

    def test_empty_portfolio_id_raises(self):
        with pytest.raises(ValueError, match="portfolio_id cannot be empty"):
            Portfolio("", initial_cash=1000.0)


class TestUpdatePrices:

    def test_update_prices_updates_market_value(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.update_prices({AAPL: 120.0})
        assert portfolio.holdings[AAPL].current_price == 120.0
        assert portfolio.holdings[AAPL].market_value == pytest.approx(1200.0)

    def test_update_prices_multiple_assets(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_buy(MSFT, quantity=5, price=200.0)
        portfolio.update_prices({AAPL: 110.0, MSFT: 210.0})
        assert portfolio.holdings[AAPL].market_value == pytest.approx(1100.0)
        assert portfolio.holdings[MSFT].market_value == pytest.approx(1050.0)

    def test_update_prices_missing_asset_leaves_stale(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_buy(MSFT, quantity=5, price=200.0)
        # Only update AAPL; MSFT stays at execution price
        portfolio.update_prices({AAPL: 110.0})
        assert portfolio.holdings[AAPL].current_price == 110.0
        assert portfolio.holdings[MSFT].current_price == 200.0  # from execute_buy

    def test_update_prices_no_holdings(self, portfolio):
        # Should not raise
        portfolio.update_prices({AAPL: 100.0})


class TestGetTotalValue:

    def test_cash_only(self, portfolio):
        assert portfolio.get_total_value() == pytest.approx(10000.0)

    def test_holdings_plus_cash(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.update_prices({AAPL: 150.0})
        # holdings: 10 * 150 = 1500, cash: 9000
        assert portfolio.get_total_value() == pytest.approx(10500.0)

    def test_after_price_drop(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.update_prices({AAPL: 80.0})
        # holdings: 10 * 80 = 800, cash: 9000
        assert portfolio.get_total_value() == pytest.approx(9800.0)

    def test_zero_price_asset(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.update_prices({AAPL: 0.0})
        # holdings: 0, cash: 9000
        assert portfolio.get_total_value() == pytest.approx(9000.0)


class TestGetWeights:

    def test_single_asset(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=100, price=100.0)
        # holdings: 100 * 100 = 10000, cash: 0
        weights = portfolio.get_weights()
        assert weights[AAPL] == pytest.approx(1.0)

    def test_two_assets_equal_value(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_buy(MSFT, quantity=10, price=100.0)
        # holdings: 1000 + 1000 = 2000, cash: 8000, total: 10000
        weights = portfolio.get_weights()
        assert weights[AAPL] == pytest.approx(0.1)
        assert weights[MSFT] == pytest.approx(0.1)

    def test_weights_include_cash_implicitly(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        # total = 10000 (1000 holdings + 9000 cash), AAPL weight = 0.1
        weights = portfolio.get_weights()
        assert weights[AAPL] == pytest.approx(0.1)

    def test_weights_zero_total_value(self):
        p = Portfolio("p1", initial_cash=0.0)
        weights = p.get_weights()
        assert weights == {}

    def test_weights_update_after_price_change(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.update_prices({AAPL: 200.0})
        # holdings: 2000, cash: 9000, total: 11000
        weights = portfolio.get_weights()
        assert weights[AAPL] == pytest.approx(2000.0 / 11000.0)


class TestGetHoldingsSummary:

    def test_cash_only_summary(self, portfolio):
        df = portfolio.get_holdings_summary()
        assert len(df) == 1
        assert df.iloc[0]["AssetID"] == "CASH"
        assert df.iloc[0]["MarketValue"] == pytest.approx(10000.0)
        assert df.iloc[0]["Weight"] == pytest.approx(1.0)

    def test_summary_with_holdings(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.update_prices({AAPL: 110.0})
        df = portfolio.get_holdings_summary()
        assert len(df) == 2  # AAPL + CASH
        aapl_row = df[df["AssetID"] == AAPL].iloc[0]
        assert aapl_row["Quantity"] == 10
        assert aapl_row["CurrentPrice"] == pytest.approx(110.0)
        assert aapl_row["MarketValue"] == pytest.approx(1100.0)

    def test_summary_weights_sum_to_one(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_buy(MSFT, quantity=5, price=200.0)
        df = portfolio.get_holdings_summary()
        assert df["Weight"].sum() == pytest.approx(1.0)


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

    def test_repr(self, portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        r = repr(portfolio)
        assert "test_portfolio" in r
        assert "num_holdings=1" in r
