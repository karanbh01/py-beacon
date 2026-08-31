# tests/test_portfolio.py
"""Unit tests for the refactored Portfolio class."""
import pandas as pd
import pytest

from beacon.exceptions import FrozenPortfolioError
from beacon.portfolio.base import Portfolio, TradeInstruction, Transaction

AAPL = "AAPL"
MSFT = "MSFT"


@pytest.fixture
def portfolio():
    return Portfolio("test_portfolio", initial_cash=10000.0)


class TestExecuteBuy:

    def test_basic_buy(self,
                       portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=150.0)
        assert AAPL in portfolio.holdings
        assert portfolio.holdings[AAPL].quantity == 10
        assert portfolio.holdings[AAPL].average_cost_price == 150.0
        assert portfolio.cash_balance == pytest.approx(8500.0)

    def test_buy_deducts_cost(self,
                              portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0, cost=50.0)
        # 10 * 100 + 50 = 1050
        assert portfolio.cash_balance == pytest.approx(10000.0 - 1050.0)

    def test_buy_records_transaction(self,
                                     portfolio):
        portfolio.execute_buy(AAPL, quantity=5, price=200.0, date=pd.Timestamp("2025-01-15"))
        assert len(portfolio.transactions) == 1
        tx = portfolio.transactions[0]
        assert tx.asset_id == AAPL
        assert tx.quantity == 5
        assert tx.price == 200.0
        assert tx.transaction_type == "BUY"
        assert tx.transaction_date == pd.Timestamp("2025-01-15")

    def test_buy_updates_market_data(self,
                                     portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=150.0)
        assert portfolio.holdings[AAPL].current_price == 150.0
        assert portfolio.holdings[AAPL].market_value == pytest.approx(1500.0)

    def test_buy_adds_to_existing_holding(self,
                                          portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_buy(AAPL, quantity=10, price=200.0)
        assert portfolio.holdings[AAPL].quantity == 20
        # avg cost = (10*100 + 10*200) / 20 = 150
        assert portfolio.holdings[AAPL].average_cost_price == pytest.approx(150.0)

    def test_insufficient_cash_skips_buy(self,
                                         portfolio):
        portfolio.execute_buy(AAPL, quantity=1000, price=100.0)  # needs 100k, only 10k
        assert AAPL not in portfolio.holdings
        assert portfolio.cash_balance == 10000.0
        assert len(portfolio.transactions) == 0

    def test_buy_zero_quantity_raises(self,
                                      portfolio):
        with pytest.raises(ValueError, match="quantity must be positive"):
            portfolio.execute_buy(AAPL, quantity=0, price=100.0)

    def test_buy_negative_quantity_raises(self,
                                          portfolio):
        with pytest.raises(ValueError, match="quantity must be positive"):
            portfolio.execute_buy(AAPL, quantity=-5, price=100.0)

    def test_buy_negative_price_raises(self,
                                       portfolio):
        with pytest.raises(ValueError, match="price cannot be negative"):
            portfolio.execute_buy(AAPL, quantity=5, price=-10.0)

    def test_buy_defaults_date_to_now(self,
                                      portfolio):
        portfolio.execute_buy(AAPL, quantity=1, price=100.0)
        assert len(portfolio.transactions) == 1
        assert isinstance(portfolio.transactions[0].transaction_date, pd.Timestamp)


class TestExecuteSell:

    def test_basic_sell(self,
                        portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_sell(AAPL, quantity=5, price=120.0)
        assert portfolio.holdings[AAPL].quantity == 5
        # cash: 10000 - 1000 + 600 = 9600
        assert portfolio.cash_balance == pytest.approx(9600.0)

    def test_sell_deducts_cost(self,
                               portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_sell(AAPL, quantity=5, price=120.0, cost=10.0)
        # cash: 10000 - 1000 + (600 - 10) = 9590
        assert portfolio.cash_balance == pytest.approx(9590.0)

    def test_sell_records_transaction(self,
                                      portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_sell(AAPL, quantity=5, price=120.0, date=pd.Timestamp("2025-02-01"))
        assert len(portfolio.transactions) == 2  # buy + sell
        tx = portfolio.transactions[1]
        assert tx.transaction_type == "SELL"
        assert tx.quantity == 5
        assert tx.price == 120.0

    def test_full_sell_removes_holding(self,
                                       portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_sell(AAPL, quantity=10, price=120.0)
        assert AAPL not in portfolio.holdings

    def test_insufficient_holdings_skips_sell(self,
                                              portfolio):
        portfolio.execute_buy(AAPL, quantity=5, price=100.0)
        initial_cash = portfolio.cash_balance
        portfolio.execute_sell(AAPL, quantity=10, price=120.0)  # only have 5
        assert portfolio.holdings[AAPL].quantity == 5  # unchanged
        assert portfolio.cash_balance == initial_cash  # unchanged
        assert len(portfolio.transactions) == 1  # only the buy

    def test_sell_asset_not_held_skips(self,
                                       portfolio):
        portfolio.execute_sell(MSFT, quantity=5, price=100.0)
        assert len(portfolio.transactions) == 0
        assert portfolio.cash_balance == 10000.0

    def test_sell_zero_quantity_raises(self,
                                       portfolio):
        with pytest.raises(ValueError, match="quantity must be positive"):
            portfolio.execute_sell(AAPL, quantity=0, price=100.0)

    def test_sell_negative_price_raises(self,
                                        portfolio):
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

    def test_update_prices_updates_market_value(self,
                                                portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.update_prices({AAPL: 120.0})
        assert portfolio.holdings[AAPL].current_price == 120.0
        assert portfolio.holdings[AAPL].market_value == pytest.approx(1200.0)

    def test_update_prices_multiple_assets(self,
                                           portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_buy(MSFT, quantity=5, price=200.0)
        portfolio.update_prices({AAPL: 110.0, MSFT: 210.0})
        assert portfolio.holdings[AAPL].market_value == pytest.approx(1100.0)
        assert portfolio.holdings[MSFT].market_value == pytest.approx(1050.0)

    def test_update_prices_missing_asset_leaves_stale(self,
                                                      portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_buy(MSFT, quantity=5, price=200.0)
        # Only update AAPL; MSFT stays at execution price
        portfolio.update_prices({AAPL: 110.0})
        assert portfolio.holdings[AAPL].current_price == 110.0
        assert portfolio.holdings[MSFT].current_price == 200.0  # from execute_buy

    def test_update_prices_no_holdings(self,
                                       portfolio):
        # Should not raise
        portfolio.update_prices({AAPL: 100.0})


class TestGetTotalValue:

    def test_cash_only(self,
                       portfolio):
        assert portfolio.get_total_value() == pytest.approx(10000.0)

    def test_holdings_plus_cash(self,
                                portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.update_prices({AAPL: 150.0})
        # holdings: 10 * 150 = 1500, cash: 9000
        assert portfolio.get_total_value() == pytest.approx(10500.0)

    def test_after_price_drop(self,
                              portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.update_prices({AAPL: 80.0})
        # holdings: 10 * 80 = 800, cash: 9000
        assert portfolio.get_total_value() == pytest.approx(9800.0)

    def test_zero_price_asset(self,
                              portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.update_prices({AAPL: 0.0})
        # holdings: 0, cash: 9000
        assert portfolio.get_total_value() == pytest.approx(9000.0)


class TestGetWeights:

    def test_single_asset(self,
                          portfolio):
        portfolio.execute_buy(AAPL, quantity=100, price=100.0)
        # holdings: 100 * 100 = 10000, cash: 0
        weights = portfolio.get_weights()
        assert weights[AAPL] == pytest.approx(1.0)

    def test_two_assets_equal_value(self,
                                    portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_buy(MSFT, quantity=10, price=100.0)
        # holdings: 1000 + 1000 = 2000, cash: 8000, total: 10000
        weights = portfolio.get_weights()
        assert weights[AAPL] == pytest.approx(0.1)
        assert weights[MSFT] == pytest.approx(0.1)

    def test_weights_include_cash_implicitly(self,
                                             portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        # total = 10000 (1000 holdings + 9000 cash), AAPL weight = 0.1
        weights = portfolio.get_weights()
        assert weights[AAPL] == pytest.approx(0.1)

    def test_weights_zero_total_value(self):
        p = Portfolio("p1", initial_cash=0.0)
        weights = p.get_weights()
        assert weights == {}

    def test_weights_update_after_price_change(self,
                                               portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.update_prices({AAPL: 200.0})
        # holdings: 2000, cash: 9000, total: 11000
        weights = portfolio.get_weights()
        assert weights[AAPL] == pytest.approx(2000.0 / 11000.0)


class TestGetHoldingsSummary:

    def test_cash_only_summary(self,
                               portfolio):
        df = portfolio.get_holdings_summary()
        assert len(df) == 1
        assert df.iloc[0]["AssetID"] == "CASH"
        assert df.iloc[0]["MarketValue"] == pytest.approx(10000.0)
        assert df.iloc[0]["Weight"] == pytest.approx(1.0)

    def test_summary_with_holdings(self,
                                   portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.update_prices({AAPL: 110.0})
        df = portfolio.get_holdings_summary()
        assert len(df) == 2  # AAPL + CASH
        aapl_row = df[df["AssetID"] == AAPL].iloc[0]
        assert aapl_row["Quantity"] == 10
        assert aapl_row["CurrentPrice"] == pytest.approx(110.0)
        assert aapl_row["MarketValue"] == pytest.approx(1100.0)

    def test_summary_weights_sum_to_one(self,
                                        portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_buy(MSFT, quantity=5, price=200.0)
        df = portfolio.get_holdings_summary()
        assert df["Weight"].sum() == pytest.approx(1.0)


class TestPortfolioIntegration:

    def test_buy_sell_cycle(self,
                            portfolio):
        """Buy, sell partial, verify state, sell rest."""
        portfolio.execute_buy(AAPL, quantity=20, price=50.0)
        assert portfolio.cash_balance == pytest.approx(9000.0)

        portfolio.execute_sell(AAPL, quantity=10, price=60.0)
        assert portfolio.cash_balance == pytest.approx(9600.0)
        assert portfolio.holdings[AAPL].quantity == 10

        portfolio.execute_sell(AAPL, quantity=10, price=70.0)
        assert portfolio.cash_balance == pytest.approx(10300.0)
        assert AAPL not in portfolio.holdings

    def test_multiple_assets(self,
                             portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_buy(MSFT, quantity=5, price=200.0)
        assert len(portfolio.holdings) == 2
        assert portfolio.cash_balance == pytest.approx(10000.0 - 1000.0 - 1000.0)

    def test_get_total_value_after_trades(self,
                                          portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.update_prices({"AAPL": 110.0})
        # holdings: 10 * 110 = 1100, cash: 9000
        assert portfolio.get_total_value() == pytest.approx(10100.0)

    def test_transactions_list_complete(self,
                                        portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        portfolio.execute_buy(MSFT, quantity=5, price=200.0)
        portfolio.execute_sell(AAPL, quantity=5, price=110.0)
        assert len(portfolio.transactions) == 3
        assert all(isinstance(tx, Transaction) for tx in portfolio.transactions)

    def test_repr(self,
                  portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=100.0)
        r = repr(portfolio)
        assert "test_portfolio" in r
        assert "num_holdings=1" in r


class TestApply:
    """BN-151: `apply(trade, date)` — the entry point for decided trades.

    The engine decides (sizing, pricing, costs); the portfolio accounts. The
    tests that matter are equivalence — a trade applied must produce exactly
    the books the old five-argument call produced — and the import geometry,
    which is most of why the move happened.
    """

    def build(self):
        from beacon.portfolio.base import Portfolio

        return Portfolio(portfolio_id="apply-test", initial_cash=100_000.0)

    def test_a_buy_applies(self):
        import pandas as pd

        from beacon.portfolio.base import TradeInstruction

        p = self.build()
        p.apply(TradeInstruction("AAA", "BUY", 100.0, 50.0, 5.0),
                pd.Timestamp("2024-01-10"))

        assert p.holdings["AAA"].quantity == 100.0
        assert p.cash_balance == 100_000.0 - 100.0 * 50.0 - 5.0

    def test_a_sell_applies(self):
        import pandas as pd

        from beacon.portfolio.base import TradeInstruction

        p = self.build()
        p.apply(TradeInstruction("AAA", "BUY", 100.0, 50.0, 0.0),
                pd.Timestamp("2024-01-10"))
        p.apply(TradeInstruction("AAA", "SELL", 40.0, 60.0, 0.0),
                pd.Timestamp("2024-02-10"))

        assert p.holdings["AAA"].quantity == 60.0

    def test_it_matches_the_direct_calls_exactly(self):
        """A second spelling, not a second accounting. Same trades through
        both paths must produce identical books."""
        import pandas as pd

        from beacon.portfolio.base import TradeInstruction

        via_apply = self.build()
        direct = self.build()
        date = pd.Timestamp("2024-01-10")

        via_apply.apply(TradeInstruction("AAA", "BUY", 100.0, 50.0, 7.5), date)
        direct.execute_buy("AAA", 100.0, 50.0, cost=7.5, date=date)

        assert via_apply.cash_balance == direct.cash_balance
        assert (via_apply.holdings["AAA"].average_cost_price
                == direct.holdings["AAA"].average_cost_price)
        assert len(via_apply.transactions) == len(direct.transactions)

    def test_an_unknown_side_is_refused(self):
        """Refused rather than guessed: silently ignoring an unknown side
        would drop a trade from the record."""
        import pytest

        from beacon.portfolio.base import TradeInstruction

        with pytest.raises(ValueError, match="Unknown trade side"):
            self.build().apply(TradeInstruction("AAA", "HOLD", 1.0, 1.0, 0.0))

    def test_the_type_is_one_class_everywhere(self):
        """The move's point: one definition in the ledger's layer, re-exported
        upward. Two classes answering the same name would make isinstance
        checks depend on the import path."""
        from beacon.backtest.engine import TradeInstruction as from_engine
        from beacon.backtest.rules import TradeInstruction as from_rules
        from beacon.portfolio.base import TradeInstruction as from_portfolio

        assert from_engine is from_portfolio
        assert from_rules is from_portfolio

    def test_the_type_checking_hack_is_gone(self):
        """BN-151's other deliverable. The engine imports BacktestModifier
        for real now; a reintroduced TYPE_CHECKING block would mean the
        cycle is back."""
        import inspect

        import beacon.backtest.engine as engine

        assert "TYPE_CHECKING" not in inspect.getsource(engine)


# ---------------------------------------------------------------------------
# BN-152: the portfolio as the store of record
# ---------------------------------------------------------------------------

INCEPTION = pd.Timestamp("2025-01-01")
POSITION_COLUMNS = ["DATE", "ASSET_ID", "QUANTITY", "PRICE", "MARKET_VALUE", "WEIGHT"]


def _booked():
    """A portfolio with a deliberately uneven history.

    Trades and marks land on distinct dates, one date carries both, and one
    name is exited entirely — so the replay, the last-write-wins rule and the
    disappearance of a closed position all have something to bite on.
    """
    p = Portfolio("booked", initial_cash=100_000.0, inception=INCEPTION)

    p.execute_buy(AAPL, quantity=100, price=150.0, date=pd.Timestamp("2025-01-02"))
    p.execute_buy(MSFT, quantity=50, price=300.0, date=pd.Timestamp("2025-01-02"))

    p.update_prices({AAPL: 160.0, MSFT: 290.0}, date=pd.Timestamp("2025-01-03"))

    # A mark and then a trade, same day.
    p.update_prices({AAPL: 155.0, MSFT: 295.0}, date=pd.Timestamp("2025-01-06"))
    p.execute_sell(AAPL, quantity=40, price=165.0, date=pd.Timestamp("2025-01-06"))

    p.update_prices({AAPL: 170.0, MSFT: 310.0}, date=pd.Timestamp("2025-01-07"))
    p.execute_sell(MSFT, quantity=50, price=305.0, date=pd.Timestamp("2025-01-08"))

    return p


def _replay(transactions,
            as_of):
    """Quantities implied by replaying every transaction up to *as_of*."""
    quantities = {}

    for tx in transactions:
        if tx.transaction_date > as_of:
            continue

        signed = tx.quantity if tx.transaction_type == "BUY" else -tx.quantity
        quantities[tx.asset_id] = quantities.get(tx.asset_id, 0.0) + signed

    # A closed position leaves the holdings, so it leaves the panel too.
    return {asset_id: qty for asset_id, qty in quantities.items() if qty > 1e-9}


def _rows_on(portfolio,
             date):
    """The recorded position rows for one date."""
    positions = portfolio.positions
    return positions[positions["DATE"] == date]


class TestInitialCapital:
    """`initial_cash` was accepted and thrown away; only the mutating balance
    survived, so a portfolio could not say what it started with."""

    def test_it_is_retained(self):
        assert Portfolio("p1", initial_cash=5000.0).initial_capital == 5000.0

    def test_it_survives_while_the_balance_moves(self,
                                                 portfolio):
        portfolio.execute_buy(AAPL, quantity=10, price=150.0)

        assert portfolio.cash_balance == pytest.approx(8500.0)
        assert portfolio.initial_capital == 10000.0

    def test_it_defaults_to_zero(self):
        assert Portfolio("p1").initial_capital == 0.0


class TestDayZero:
    """With an inception date the books open at what they were given, before
    anything trades."""

    def test_nav_opens_at_the_initial_capital(self):
        p = Portfolio("p1", initial_cash=100_000.0, inception=INCEPTION)

        assert p.nav.index[0] == INCEPTION
        assert p.nav.iloc[0] == pytest.approx(100_000.0)

    def test_cash_opens_at_the_initial_capital(self):
        p = Portfolio("p1", initial_cash=100_000.0, inception=INCEPTION)

        assert p.cash.index[0] == INCEPTION
        assert p.cash.iloc[0] == pytest.approx(100_000.0)

    def test_day_zero_holds_nothing(self):
        p = Portfolio("p1", initial_cash=100_000.0, inception=INCEPTION)

        assert _rows_on(p, INCEPTION).empty

    def test_it_stays_the_first_row_once_trading_starts(self):
        p = _booked()

        assert p.nav.index[0] == INCEPTION
        assert p.nav.iloc[0] == pytest.approx(100_000.0)

    def test_without_inception_history_starts_at_the_first_event(self):
        """A hand-built portfolio need not name a day zero."""
        p = Portfolio("hand-built", initial_cash=10_000.0)

        assert p.nav.empty

        p.execute_buy(AAPL, quantity=10, price=100.0, date=pd.Timestamp("2025-03-04"))

        assert list(p.nav.index) == [pd.Timestamp("2025-03-04")]
        assert p.nav.iloc[0] == pytest.approx(10_000.0)


class TestHistoryPanels:
    """Positions, cash and NAV at rest: plain frames and series, no classes."""

    def test_positions_are_long_form(self):
        assert list(_booked().positions.columns) == POSITION_COLUMNS

    def test_a_row_per_held_asset_per_recorded_date(self):
        rows = _rows_on(_booked(), pd.Timestamp("2025-01-03"))

        assert sorted(rows["ASSET_ID"]) == [AAPL, MSFT]

    def test_a_mark_is_recorded_like_a_trade(self):
        """Nothing traded on the 3rd; the marks alone are an event."""
        rows = _rows_on(_booked(), pd.Timestamp("2025-01-03"))
        aapl = rows[rows["ASSET_ID"] == AAPL].iloc[0]

        assert aapl["QUANTITY"] == pytest.approx(100.0)
        assert aapl["PRICE"] == pytest.approx(160.0)
        assert aapl["MARKET_VALUE"] == pytest.approx(16_000.0)

    def test_the_last_write_on_a_date_wins(self):
        """The 6th carries a mark and then a sell; the day ends post-trade."""
        p = _booked()
        rows = _rows_on(p, pd.Timestamp("2025-01-06"))
        aapl = rows[rows["ASSET_ID"] == AAPL].iloc[0]

        assert aapl["QUANTITY"] == pytest.approx(60.0)
        assert aapl["PRICE"] == pytest.approx(165.0)
        assert p.cash.loc[pd.Timestamp("2025-01-06")] == pytest.approx(
            100_000.0 - 15_000.0 - 15_000.0 + 40 * 165.0)

    def test_a_closed_position_leaves_the_panel(self):
        rows = _rows_on(_booked(), pd.Timestamp("2025-01-08"))

        assert list(rows["ASSET_ID"]) == [AAPL]

    def test_cash_and_nav_are_date_indexed_series(self):
        p = _booked()

        assert isinstance(p.cash, pd.Series)
        assert isinstance(p.nav, pd.Series)
        assert list(p.cash.index) == list(p.nav.index)
        assert p.nav.index.is_monotonic_increasing

    def test_an_untouched_portfolio_is_empty_but_typed(self):
        """Empty is not a reason to hand back untyped columns: a caller
        filtering on DATE must not have to special-case the empty run."""
        p = Portfolio("never-traded")

        assert list(p.positions.columns) == POSITION_COLUMNS
        assert p.positions.empty
        assert p.positions["DATE"].dtype.kind == "M"
        assert p.positions["QUANTITY"].dtype == "float64"
        assert p.cash.empty and p.cash.dtype == "float64"
        assert p.nav.empty and p.nav.dtype == "float64"


class TestReconciliation:
    """The pinned invariant. Positions and transactions are two lenses on one
    history, and the double-counting is only safe while they agree."""

    def test_positions_replay_the_transaction_log(self):
        p = _booked()

        for as_of in p.nav.index:
            rows = _rows_on(p, as_of)
            recorded = dict(zip(rows["ASSET_ID"], rows["QUANTITY"], strict=True))

            assert recorded == pytest.approx(_replay(p.transactions, as_of)), as_of

    def test_a_mid_run_date_reconciles_too(self):
        """Named separately from the sweep above: the interesting failure is a
        partial sell, and a bug there is easy to lose in a loop's message."""
        p = _booked()
        as_of = pd.Timestamp("2025-01-06")

        assert _replay(p.transactions, as_of) == {AAPL: pytest.approx(60.0),
                                                  MSFT: pytest.approx(50.0)}

        rows = _rows_on(p, as_of)
        recorded = dict(zip(rows["ASSET_ID"], rows["QUANTITY"], strict=True))

        assert recorded == pytest.approx(_replay(p.transactions, as_of))

    def test_the_stored_weight_is_the_recorded_value_over_the_recorded_nav(self):
        p = _booked()
        nav = p.nav

        for _, row in p.positions.iterrows():
            assert row["WEIGHT"] == pytest.approx(
                row["MARKET_VALUE"] / nav.loc[row["DATE"]]), row["DATE"]

    def test_nav_is_the_recorded_positions_plus_the_recorded_cash(self):
        p = _booked()
        held = p.positions.groupby("DATE")["MARKET_VALUE"].sum()

        for date, nav in p.nav.items():
            assert nav == pytest.approx(held.get(date, 0.0) + p.cash.loc[date]), date


class TestFreeze:
    """After a run the portfolio is the record of it, and a later write would
    restate a result somebody has already read."""

    def _frozen(self):
        p = _booked()
        p.freeze()
        return p

    def test_a_fresh_portfolio_is_not_frozen(self):
        assert Portfolio("p1").frozen is False

    def test_apply_is_refused(self):
        with pytest.raises(FrozenPortfolioError):
            self._frozen().apply(TradeInstruction(AAPL, "BUY", 1.0, 100.0, 0.0),
                                 pd.Timestamp("2025-01-09"))

    def test_execute_buy_is_refused(self):
        with pytest.raises(FrozenPortfolioError):
            self._frozen().execute_buy(AAPL, quantity=1, price=100.0)

    def test_execute_sell_is_refused(self):
        with pytest.raises(FrozenPortfolioError):
            self._frozen().execute_sell(AAPL, quantity=1, price=100.0)

    def test_update_prices_is_refused(self):
        with pytest.raises(FrozenPortfolioError):
            self._frozen().update_prices({AAPL: 100.0})

    def test_the_message_says_what_to_do_instead(self):
        with pytest.raises(FrozenPortfolioError, match="Seed a new run"):
            self._frozen().execute_buy(AAPL, quantity=1, price=100.0)

    def test_the_error_names_the_portfolio_and_the_operation(self):
        with pytest.raises(FrozenPortfolioError) as raised:
            self._frozen().update_prices({AAPL: 100.0})

        assert raised.value.portfolio_id == "booked"
        assert raised.value.operation == "update_prices"

    def test_reads_still_work(self):
        p = self._frozen()

        assert not p.positions.empty
        assert p.nav.iloc[0] == pytest.approx(100_000.0)
        assert p.get_total_value() > 0

    def test_freezing_twice_is_harmless(self):
        p = self._frozen()
        p.freeze()

        assert p.frozen is True
