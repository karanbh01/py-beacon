# tests/test_portfolio_asset_view.py
"""BN-156: `p.asset("AAA")` — position and market data from one object.

Two things carry the weight here: the resolution order (bound wins, ambient
falls back, and the error names both fixes), and that everything is read
live — nothing cached on the view can go stale.
"""
import logging

import pandas as pd
import pytest

import beacon
from beacon import sources
from beacon.exceptions import DataSourceError
from beacon.portfolio.asset_view import PortfolioAssetView
from beacon.portfolio.base import Portfolio, TradeInstruction
from beacon.testing import dataset


@pytest.fixture(autouse=True)
def clean_ambient():
    """Every test starts and ends with no process-level source."""
    sources._reset_for_tests()
    yield
    sources._reset_for_tests()


def _bought(source=None):
    """A portfolio holding 100 AAA at 50, marked to 61.5."""
    p = Portfolio(portfolio_id="view-test", initial_cash=1_000_000.0,
                  source=source)
    p.apply(TradeInstruction("AAA", "BUY", 100.0, 50.0, 0.0),
            pd.Timestamp("2024-06-03"))
    p.update_prices({"AAA": 61.5}, pd.Timestamp("2024-06-04"))
    return p


class TestThePositionNumbers:
    """Read from the books, live."""

    def test_quantity_and_cost(self):
        view = _bought(dataset.data_fetcher()).asset("AAA")

        assert view.quantity == 100.0
        assert view.average_cost == 50.0

    def test_market_value_and_weight(self):
        p = _bought(dataset.data_fetcher())
        view = p.asset("AAA")

        assert view.market_value == pytest.approx(6150.0)
        assert view.weight == pytest.approx(6150.0 / p.get_total_value())

    def test_unrealised_pnl(self):
        view = _bought(dataset.data_fetcher()).asset("AAA")

        assert view.unrealised_pnl == pytest.approx((61.5 - 50.0) * 100.0)

    def test_the_view_is_live_not_cached(self):
        """A later mark shows through an existing view: the view is a lens
        over the books, not a copy of them."""
        p = _bought(dataset.data_fetcher())
        view = p.asset("AAA")

        before = view.market_value
        p.update_prices({"AAA": 70.0}, pd.Timestamp("2024-06-05"))

        assert before == pytest.approx(6150.0)
        assert view.market_value == pytest.approx(7000.0)

    def test_a_sold_out_position_reads_as_zero_not_missing(self):
        """Sold out is still in the books — the asset was held, so the view
        answers, with a quantity of zero and no P&L to claim."""
        p = _bought(dataset.data_fetcher())
        p.apply(TradeInstruction("AAA", "SELL", 100.0, 65.0, 0.0),
                pd.Timestamp("2024-06-05"))
        view = p.asset("AAA")

        assert view.quantity == 0.0
        assert view.unrealised_pnl is None
        assert not view.position_history().empty

    def test_a_never_held_asset_raises(self):
        with pytest.raises(KeyError, match="does not appear"):
            _bought(dataset.data_fetcher()).asset("NOSUCH")


class TestTheMarketSide:
    """Read from the data source, through the inherited surface."""

    def test_prices_flow_through(self):
        view = _bought(dataset.data_fetcher()).asset("AAA")
        prices = view.prices("2024-06-01", "2024-06-30")

        assert not prices.empty
        assert "CLOSE" in prices.columns

    def test_sector_is_point_in_time(self):
        view = _bought(dataset.data_fetcher()).asset("AAA")

        assert view.sector(on="2024-06-03") == "Technology"

    def test_repr_reads(self):
        view = _bought(dataset.data_fetcher()).asset("AAA")

        assert "AAA" in repr(view)
        assert "100" in repr(view)


class TestResolutionOrder:
    """Bound wins; ambient falls back; the error names both fixes."""

    def test_a_bound_source_wins_over_the_ambient_one(self):
        """A result analysed next week must read its run's data, however the
        process source has moved since."""
        bound = dataset.data_fetcher()
        p = _bought(source=bound)

        other = dataset.data_fetcher()
        beacon.use(other)

        view = p.asset("AAA")

        assert view._data_fetcher is bound
        assert view._data_fetcher is not other

    def test_an_unbound_portfolio_uses_the_process_source(self):
        fetcher = dataset.data_fetcher()
        beacon.use(fetcher)

        view = _bought().asset("AAA")

        assert view._data_fetcher is fetcher

    def test_use_none_clears(self,
                             tmp_path,
                             monkeypatch):
        """Cleared means back to the fallback chain — and with the default
        store pointed at nothing (this machine has a real one), the chain
        ends in the error."""
        from beacon.data import store

        monkeypatch.setattr(store, "default_path",
                            lambda: tmp_path / "nothing")
        beacon.use(dataset.data_fetcher())
        beacon.use(None)

        with pytest.raises(DataSourceError):
            _bought().asset("AAA")

    def test_the_default_store_is_the_fallback(self,
                                               tmp_path,
                                               monkeypatch):
        """No binding, no beacon.use — the store the generator writes and the
        server auto-loads answers, loaded lazily and cached."""
        from beacon.data import store

        logging.disable(logging.ERROR)
        try:
            store.save(dataset.data_fetcher(), tmp_path / "store")
        finally:
            logging.disable(logging.NOTSET)

        monkeypatch.setattr(store, "default_path",
                            lambda: tmp_path / "store")

        view = _bought().asset("AAA")

        assert not view.prices("2024-06-01", "2024-06-30").empty

        # Cached: a second resolve is the same object, not a second load.
        assert sources.resolve() is sources.resolve()

    def test_no_source_anywhere_names_both_fixes(self,
                                                 tmp_path,
                                                 monkeypatch):
        """"No data" deep inside a price lookup is useless unless it says
        what to do about it."""
        from beacon.data import store

        monkeypatch.setattr(store, "default_path",
                            lambda: tmp_path / "nothing")

        with pytest.raises(DataSourceError) as raised:
            _bought().asset("AAA")

        message = str(raised.value)

        assert "beacon.use" in message
        assert "beacon.synthetic" in message

    def test_use_is_exported_from_the_package(self):
        assert beacon.use is sources.use


class TestTheEngineBindsItsRun:
    """A backtest result's views read the data the run used, with no setup."""

    def _run(self):
        from unittest.mock import MagicMock

        from beacon.backtest.engine import BacktestEngine

        dates = pd.bdate_range("2025-01-02", periods=5, freq="B")
        dp = MagicMock()

        def fetch(asset_id, start, end, columns=None):
            index = pd.DatetimeIndex([d for d in dates
                                      if str(start) <= str(d.date()) <= str(end)])
            return pd.DataFrame({"CLOSE": [100.0] * len(index)}, index=index)

        dp.fetch_market_data.side_effect = fetch
        dp.fetch_reference_data.return_value = pd.DataFrame()
        dp.fetch_fx_rates.return_value = pd.Series(dtype=float)
        dp.delisting_dates = {}

        engine = BacktestEngine(str(dates[0].date()), str(dates[-1].date()),
                                10_000.0, dp,
                                target_weights={dates[0]: {"AAA": 1.0}})
        return engine.run(), dp

    def test_the_portfolio_is_bound_to_the_run_data(self):
        result, dp = self._run()

        assert result.portfolio.source is dp
        # And .asset() works with nothing ambient set at all.
        view = result.portfolio.asset("AAA")
        assert isinstance(view, PortfolioAssetView)

    def test_the_result_is_bound_too(self):
        """`result.asset()` needed `with_data` before; the engine now binds
        on the way out, so the parity with IndexCalculator holds."""
        result, dp = self._run()

        assert result._data_fetcher is dp
        assert result.asset("AAA").asset_id == "AAA"
