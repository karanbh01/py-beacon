# tests/test_backtest_result.py
"""Unit tests for BacktestResult, Book and BacktestAssetView.

Rewritten for BN-154: the result holds books rather than flattened fields.
Fixtures build a real Portfolio and feed its history recorder directly — the
recorder is the write path the engine uses, so the fixtures exercise the same
plumbing while pinning exact NAV and weight paths the way the old flat
fixtures did.
"""
from unittest.mock import MagicMock

import pandas as pd
import pytest

from beacon.backtest.asset_view import BacktestAssetView
from beacon.backtest.result import BacktestResult, Book
from beacon.index.result import IndexResult
from beacon.portfolio.base import Holding, Portfolio, Transaction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

START = "2025-01-02"


def _dates(count,
           start=START):
    return pd.bdate_range(start=start, periods=count, freq="B")


def _make_portfolio(nav_values,
                    weights_per_asset=None,
                    txns=(),
                    initial=10000.0,
                    start=START):
    """A portfolio whose recorded history follows the given paths exactly.

    Rows are written through the history recorder — the same write path the
    engine drives — with holdings shaped so each asset's stored weight equals
    the requested one and the recorded NAV equals the requested value. An
    asset with weight 0 on a date is simply absent that day, which is what a
    sold-out position looks like in real books.
    """
    weights_per_asset = weights_per_asset or {}
    dates = _dates(len(nav_values), start)
    eve = dates[0] - pd.tseries.offsets.BDay(1) if len(dates) else None

    portfolio = Portfolio(portfolio_id="test_bt",
                          initial_cash=initial,
                          inception=eve)

    for position, date in enumerate(dates):
        nav = float(nav_values[position])
        holdings = {}

        for asset_id, weights in weights_per_asset.items():
            weight = weights[position]

            if weight <= 0:
                continue

            value = weight * nav
            holdings[asset_id] = Holding(asset_id=asset_id,
                                         quantity=1.0,
                                         average_cost_price=value,
                                         current_price=value,
                                         market_value=value)

        cash = nav - sum(h.market_value for h in holdings.values())
        portfolio._history.record(date, holdings, cash)

    portfolio.transactions.extend(txns)

    return portfolio


def _make_result(nav_values=None,
                 weights_per_asset=None,
                 txns=(),
                 index_levels=None,
                 **overrides):
    """Create a BacktestResult with sensible defaults."""
    nav_values = (nav_values if nav_values is not None
                  else [10000, 10100, 10200, 10150, 10300])
    weights_per_asset = weights_per_asset if weights_per_asset is not None else {
        "AAPL": [0.3, 0.31, 0.32, 0.31, 0.33],
        "MSFT": [0.2, 0.19, 0.18, 0.19, 0.17],
    }

    if not nav_values:
        weights_per_asset = {}

    portfolio = _make_portfolio(nav_values, weights_per_asset, txns)

    fields = {"portfolio": portfolio}
    if index_levels is not None:
        fields["index"] = Book.from_index(_make_index_result(index_levels))
    fields.update(overrides)

    return BacktestResult(**fields)


def _make_index_result(values,
                       weight_snapshots=None):
    """Create a minimal IndexResult for tracking tests."""
    levels = pd.Series([float(v) for v in values], index=_dates(len(values)))
    return IndexResult(
        index_id="target_idx",
        index_levels=levels,
        divisor_history=pd.Series(1.0, index=levels.index),
        constituent_snapshots={},
        weight_snapshots=weight_snapshots or {},
    )


# ---------------------------------------------------------------------------
# BacktestResult tests
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_fields(self):
        r = _make_result()
        assert r.portfolio.portfolio_id == "test_bt"
        assert r.portfolio.initial_capital == 10000.0
        assert len(r.trading_nav) == 5
        assert r.index is None
        assert r.benchmark is None
        assert r.target_index is None
        assert r._data_fetcher is None

    def test_nav_opens_with_the_capital(self):
        """The portfolio's own NAV carries the day-zero row; the trading NAV
        excludes it, so metrics see exactly the series they always saw."""
        r = _make_result()

        assert r.portfolio.nav.iloc[0] == 10000.0
        assert len(r.portfolio.nav) == 6
        assert r.trading_nav.iloc[0] == 10000.0
        assert r.trading_nav.index[0] == _dates(1)[0]

    def test_repr(self):
        r = _make_result()
        s = repr(r)
        assert "test_bt" in s
        assert "dates=5" in s
        assert "transactions=0" in s
        assert "index=False" in s
        assert "benchmark=False" in s
        assert "data_bound=False" in s

    def test_repr_with_index_and_data(self):
        r = _make_result(index_levels=[100, 101, 102, 103, 104])
        r.with_data(MagicMock())
        s = repr(r)
        assert "index=True" in s
        assert "data_bound=True" in s

    def test_data_fetcher_excluded_from_repr(self):
        r = _make_result()
        r.with_data(MagicMock())
        assert "_data_fetcher" not in repr(r)

    def test_the_old_flat_fields_are_gone(self):
        """One name, one home (decision 15). The redesign deleted the flat
        fields and the alias; anything still reading them should fail loudly
        here rather than in a downstream consumer."""
        r = _make_result()

        for name in ("portfolio_nav", "cash_history", "actual_weight_history",
                     "portfolio_id", "transactions", "initial_capital"):
            assert not hasattr(r, name), name


class TestBook:

    def test_from_index_carries_levels_and_source(self):
        idx = _make_index_result([100, 101, 102])
        book = Book.from_index(idx)

        assert book.levels.equals(idx.index_levels)
        assert book.source is idx
        assert book.weights.empty  # no daily panel on this fixture

    def test_from_levels_has_no_weights(self):
        """A benchmark from a bare series has no composition; the frame is
        empty rather than invented."""
        book = Book.from_levels(pd.Series([1.0, 1.1], index=_dates(2)))

        assert book.weights.empty
        assert book.source is None

    def test_returns(self):
        book = Book.from_levels(pd.Series([100.0, 110.0], index=_dates(2)))

        assert book.returns.iloc[0] == pytest.approx(0.10)

    def test_daily_panel_pivots_wide(self):
        idx = _make_index_result([100, 101])
        idx.daily_weights = pd.DataFrame({
            "DATE": list(_dates(2)) * 2,
            "IDENTIFIER": ["AAA", "AAA", "BBB", "BBB"],
            "AMOUNT": [1.0, 1.0, 2.0, 2.0],
            "WEIGHT": [0.6, 0.55, 0.4, 0.45],
        })
        book = Book.from_index(idx)

        assert book.weights.loc[_dates(2)[1], "BBB"] == pytest.approx(0.45)


class TestWithData:

    def test_returns_self(self):
        r = _make_result()
        fetcher = MagicMock()
        assert r.with_data(fetcher) is r

    def test_binds_fetcher(self):
        r = _make_result()
        fetcher = MagicMock()
        r.with_data(fetcher)
        assert r._data_fetcher is fetcher


class TestAsset:

    def test_raises_without_data(self):
        r = _make_result()
        with pytest.raises(RuntimeError, match="No DataFetcher bound"):
            r.asset("AAPL")

    def test_raises_for_unknown_asset(self):
        r = _make_result()
        r.with_data(MagicMock())
        with pytest.raises(KeyError, match="does not appear"):
            r.asset("UNKNOWN")

    def test_returns_backtest_asset_view(self):
        r = _make_result()
        r.with_data(MagicMock())
        view = r.asset("AAPL")
        assert isinstance(view, BacktestAssetView)
        assert view.asset_id == "AAPL"


class TestGetReturns:

    def test_returns_series(self):
        r = _make_result()
        returns = r.get_returns()
        assert len(returns) == 4  # 5 NAV values -> 4 returns
        expected_first = (10100 / 10000) - 1
        assert returns.iloc[0] == pytest.approx(expected_first)

    def test_the_day_zero_row_is_not_a_return(self):
        """The eve row records what the run started with; treating it as a
        traded day would add a synthetic first return and move every
        annualised metric."""
        r = _make_result()

        assert len(r.portfolio.nav) == 6
        assert len(r.get_returns()) == 4

    def test_empty_nav(self):
        r = _make_result(nav_values=[])
        returns = r.get_returns()
        assert returns.empty


class TestAgainst:

    def test_it_computes_relative_metrics(self):
        r = _make_result()
        metrics = r.against(_make_index_result([10000, 10100, 10200, 10150,
                                                10300]))

        assert metrics.excess_return == pytest.approx(0.0, abs=1e-12)

    def test_it_accepts_a_bare_series(self):
        r = _make_result()
        series = pd.Series([10000.0, 10050, 10100, 10000, 10200],
                           index=_dates(5))

        assert r.against(series) is not None

    def test_it_accepts_another_result(self):
        r = _make_result()
        other = _make_result(nav_values=[10000, 10050, 10150, 10100, 10250])

        assert r.against(other) is not None

    def test_it_never_mutates_the_record(self):
        """The exploratory half of decision 13: questions must not edit
        facts. Ten comparisons leave the stored books untouched."""
        r = _make_result(index_levels=[10000, 10100, 10200, 10150, 10300])
        benchmark_before = r.benchmark
        nav_before = r.portfolio.nav.copy()

        for _ in range(10):
            r.against(pd.Series([10000.0, 10100, 10150, 10100, 10250],
                                index=_dates(5)))

        assert r.benchmark is benchmark_before
        assert r.portfolio.nav.equals(nav_before)

    def test_an_unusable_comparator_is_refused(self):
        with pytest.raises(TypeError, match="Cannot compare"):
            _make_result().against(42)  # type: ignore[arg-type]


class TestTrackingError:

    def test_none_without_index(self):
        r = _make_result()
        assert r.get_tracking_error() is None

    def test_zero_for_identical_series(self):
        nav_values = [10000, 10100, 10200, 10150, 10300]
        r = _make_result(index_levels=nav_values)
        te = r.get_tracking_error()
        assert te == pytest.approx(0.0, abs=1e-10)

    def test_positive_for_divergent_series(self):
        r = _make_result(index_levels=[10000, 10200, 10400, 10100, 10500])
        te = r.get_tracking_error()
        assert te is not None
        assert te > 0


class TestTrackingDifference:

    def test_none_without_index(self):
        r = _make_result()
        assert r.get_tracking_difference() is None

    def test_zero_for_identical_series(self):
        nav_values = [10000, 10100, 10200, 10150, 10300]
        r = _make_result(index_levels=nav_values)
        td = r.get_tracking_difference()
        assert td == pytest.approx(0.0, abs=1e-10)

    def test_positive_when_portfolio_outperforms(self):
        # Portfolio gains more than index (NAV ends at 10300)
        r = _make_result(index_levels=[10000, 10050, 10100, 10050, 10100])
        td = r.get_tracking_difference()
        assert td is not None
        assert td > 0

    def test_negative_when_portfolio_underperforms(self):
        r = _make_result(index_levels=[10000, 10200, 10500, 10800, 11000])
        td = r.get_tracking_difference()
        assert td is not None
        assert td < 0


class TestSummary:

    def test_keys_without_index(self):
        r = _make_result()
        s = r.summary()
        assert "total_return" in s
        assert "annualised_return" in s
        assert "volatility" in s
        assert "sharpe_ratio" in s
        assert "max_drawdown" in s
        assert "tracking_error" not in s
        assert "tracking_difference" not in s

    def test_keys_with_index(self):
        r = _make_result(index_levels=[10000, 10100, 10200, 10150, 10300])
        s = r.summary()
        assert "tracking_error" in s
        assert "tracking_difference" in s

    def test_total_return(self):
        r = _make_result()
        s = r.summary()
        # 10300 / 10000 - 1 = 0.03
        assert s["total_return"] == pytest.approx(0.03)

    def test_max_drawdown(self):
        # NAV: 10000, 10100, 10200, 10150, 10300
        # Drawdown from peak 10200 to 10150 = (10150-10200)/10200
        r = _make_result()
        s = r.summary()
        expected_dd = (10150 - 10200) / 10200
        assert s["max_drawdown"] == pytest.approx(expected_dd)

    def test_volatility_positive(self):
        r = _make_result()
        s = r.summary()
        assert s["volatility"] > 0

    def test_empty_nav_summary(self):
        r = _make_result(nav_values=[])
        s = r.summary()
        assert s["total_return"] == 0.0
        assert s["volatility"] == 0.0


# ---------------------------------------------------------------------------
# BacktestAssetView tests
# ---------------------------------------------------------------------------

DATES_5 = pd.bdate_range(start=START, periods=5, freq="B")

_SAMPLE_TXNS = [
    Transaction("AAPL", 10, 150.0, "BUY", DATES_5[0], 5.0),
    Transaction("AAPL", 5, 160.0, "SELL", DATES_5[2], 3.0),
    Transaction("MSFT", 20, 300.0, "BUY", DATES_5[1], 10.0),
]


def _make_view(asset_id="AAPL",
               weights_per_asset=None,
               txns=(),
               target_snapshots=None,
               nav_values=None):
    """A view over a portfolio whose books follow the given weight paths."""
    weights_per_asset = (weights_per_asset if weights_per_asset is not None
                         else {"AAPL": [0.3, 0.31, 0.32, 0.0, 0.33]})
    count = len(next(iter(weights_per_asset.values())))
    nav_values = nav_values or [10000.0] * count

    portfolio = _make_portfolio(nav_values, weights_per_asset, txns)

    index_book = None
    if target_snapshots is not None:
        index_book = Book.from_index(
            _make_index_result([100.0] * count,
                               weight_snapshots=target_snapshots))

    return BacktestAssetView(asset_id, MagicMock(), portfolio, index_book)


class TestBacktestAssetViewWeights:

    def test_repr(self):
        view = _make_view()
        assert "BacktestAssetView" in repr(view)
        assert "AAPL" in repr(view)

    def test_weight_series_excludes_the_unheld_day(self):
        view = _make_view()
        ws = view.weight_series()
        assert len(ws) == 4  # sold out on the fourth date
        assert all(w > 0 for w in ws)

    def test_the_alias_is_gone(self):
        """`actual_weight_series` was a one-line alias of `weight_series` —
        two names for one series. It died with the redesign, and a test pins
        that so it cannot quietly come back."""
        assert not hasattr(_make_view(), "actual_weight_series")

    def test_weight_on_date_returns_value(self):
        view = _make_view()
        w = view.weight_on_date(DATES_5[0])
        assert w == pytest.approx(0.3)

    def test_weight_on_date_returns_none_when_not_held(self):
        """The books were written that day and the asset has no row: it was
        not held, and falling back to an earlier weight would report a
        position that had already been sold."""
        view = _make_view()
        w = view.weight_on_date(DATES_5[3])
        assert w is None

    def test_weight_on_date_falls_back_off_calendar(self):
        """A weekend is not a run day; the position in force is the last
        recorded one."""
        view = _make_view()
        saturday = DATES_5[1] + pd.Timedelta(days=1)
        assert saturday.dayofweek == 5
        assert saturday not in view._calendar()
        assert view.weight_on_date(saturday) == pytest.approx(0.31)

    def test_weight_on_date_returns_none_before_history(self):
        view = _make_view()
        w = view.weight_on_date(pd.Timestamp("2020-01-01"))
        assert w is None

    def test_weight_series_unknown_asset(self):
        view = _make_view(asset_id="UNKNOWN",
                          weights_per_asset={"AAPL": [0.3, 0.31]})
        assert view.weight_series().empty


class TestBacktestAssetViewTrades:

    def test_trades_filters_by_asset(self):
        view = _make_view(txns=_SAMPLE_TXNS)
        df = view.trades()
        assert len(df) == 2  # only AAPL txns
        assert set(df["type"]) == {"BUY", "SELL"}

    def test_trades_columns(self):
        view = _make_view(txns=_SAMPLE_TXNS)
        df = view.trades()
        assert list(df.columns) == ["date", "type", "quantity", "price", "cost"]

    def test_trades_empty_for_no_trades(self):
        view = _make_view(asset_id="GOOG",
                          weights_per_asset={"GOOG": [0.1, 0.1]},
                          txns=_SAMPLE_TXNS)
        df = view.trades()
        assert df.empty
        assert list(df.columns) == ["date", "type", "quantity", "price", "cost"]

    def test_trades_values(self):
        view = _make_view(txns=_SAMPLE_TXNS)
        df = view.trades()
        buy = df[df["type"] == "BUY"].iloc[0]
        assert buy["quantity"] == 10
        assert buy["price"] == 150.0
        assert buy["cost"] == 5.0


class TestBacktestAssetViewTotalCost:

    def test_total_cost(self):
        view = _make_view(txns=_SAMPLE_TXNS)
        assert view.total_cost() == pytest.approx(8.0)  # 5.0 + 3.0

    def test_total_cost_zero_for_other_asset(self):
        view = _make_view(asset_id="GOOG",
                          weights_per_asset={"GOOG": [0.1]},
                          txns=_SAMPLE_TXNS)
        assert view.total_cost() == 0.0


class TestBacktestAssetViewHoldingPeriods:

    def test_single_continuous_period(self):
        view = _make_view(weights_per_asset={"AAPL": [0.3, 0.31, 0.32]})
        periods = view.holding_periods()
        assert len(periods) == 1
        dates = _dates(3)
        assert periods[0]["start"] == dates[0]
        assert periods[0]["end"] == dates[2]

    def test_gap_creates_two_periods(self):
        """Sold out and re-bought is two periods, not one — the panel has no
        rows on the unheld days, and the run calendar is what makes the gap
        visible."""
        view = _make_view(
            weights_per_asset={"AAPL": [0.3, 0.31, 0.0, 0.0, 0.33]})
        periods = view.holding_periods()
        assert len(periods) == 2

    def test_no_holdings(self):
        view = _make_view(weights_per_asset={"AAPL": [0.0, 0.0, 0.0]})
        assert view.holding_periods() == []

    def test_unknown_asset_no_periods(self):
        view = _make_view(asset_id="UNKNOWN",
                          weights_per_asset={"AAPL": [0.3]})
        assert view.holding_periods() == []


class TestBacktestAssetViewTargetWeights:

    def _target_snapshots(self):
        dates = _dates(5)
        return {
            dates[0]: {"AAPL": 0.30, "MSFT": 0.20},
            dates[2]: {"AAPL": 0.30, "MSFT": 0.20},
        }

    def _view_with_target(self):
        return _make_view(
            weights_per_asset={"AAPL": [0.30, 0.32, 0.28, 0.31, 0.29]},
            target_snapshots=self._target_snapshots())

    def test_target_weight_series(self):
        view = self._view_with_target()
        ts = view.target_weight_series()
        assert len(ts) == 2
        assert all(w == pytest.approx(0.30) for w in ts)

    def test_target_weight_series_empty_without_target(self):
        view = _make_view(weights_per_asset={"AAPL": [0.3]})
        assert view.target_weight_series().empty

    def test_slippage_vs_target(self):
        view = self._view_with_target()
        slip = view.slippage_vs_target()
        assert not slip.empty
        # First date: actual 0.30 - target 0.30 = 0.0
        assert slip.iloc[0] == pytest.approx(0.0)
        # Second date: actual 0.32 - target 0.30 = 0.02
        assert slip.iloc[1] == pytest.approx(0.02)

    def test_slippage_empty_without_target(self):
        view = _make_view(weights_per_asset={"AAPL": [0.3]})
        assert view.slippage_vs_target().empty

    def test_slippage_negative_when_underweight(self):
        view = self._view_with_target()
        slip = view.slippage_vs_target()
        # Third date: actual 0.28 - target 0.30 = -0.02
        assert slip.iloc[2] == pytest.approx(-0.02)


class TestBacktestResultAssetIntegration:

    def test_asset_reads_the_books_and_the_index(self):
        dates = _dates(3)
        target_idx = IndexResult(
            index_id="idx",
            index_levels=pd.Series([100.0, 101, 102], index=dates),
            divisor_history=pd.Series([1.0, 1.0, 1.0], index=dates),
            constituent_snapshots={dates[0]: ["AAPL"]},
            weight_snapshots={dates[0]: {"AAPL": 0.5}},
        )
        txns = [Transaction("AAPL", 10, 100.0, "BUY", dates[0], 2.0)]
        r = _make_result(nav_values=[10000, 10050, 10100],
                         weights_per_asset={"AAPL": [0.5, 0.51, 0.49]},
                         txns=txns,
                         index=Book.from_index(target_idx))
        r.with_data(MagicMock())
        view = r.asset("AAPL")
        assert isinstance(view, BacktestAssetView)
        assert len(view.trades()) == 1
        assert view.total_cost() == pytest.approx(2.0)
        assert not view.target_weight_series().empty
        assert not view.slippage_vs_target().empty
