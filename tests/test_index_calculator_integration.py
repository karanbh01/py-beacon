# tests/test_index_calculator_integration.py
"""Integration test for IndexCalculator.run() with synthetic market data."""
from unittest.mock import MagicMock

import pandas as pd
import pytest

from beacon.asset.equity import Equity
from beacon.exceptions import CalculationError
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.index.result import IndexResult
from beacon.index.schedule import sessions

# ---------------------------------------------------------------------------
# Synthetic universe
# ---------------------------------------------------------------------------
ASSET_A = Equity(name="Asset A", currency="USD", ticker="ASSET_A", exchange="NYSE")
ASSET_B = Equity(name="Asset B", currency="USD", ticker="ASSET_B", exchange="NYSE")

BASE_DATE = "2024-01-02"  # First session of Jan 2024
END_DATE = "2024-03-29"   # ~3 months of data
CALENDAR = "XNYS"
BASE_VALUE = 1000.0
SHARES = 1000  # Fixed shares outstanding for both assets


def _build_price_series():
    """Build deterministic daily prices for ASSET_A and ASSET_B.

    ASSET_A: starts at 100, gains 10% over the full period (linear daily).
    ASSET_B: starts at 200, gains 10% over the full period (linear daily).

    Returns a dict mapping (ticker, date_str) -> price.
    """
    trading_days = pd.bdate_range(start=BASE_DATE, end=END_DATE)
    n = len(trading_days)

    prices = {}
    for i, day in enumerate(trading_days):
        date_str = day.strftime("%Y-%m-%d")
        # Linear interpolation: start * (1 + 0.10 * i/(n-1))
        price_a = 100.0 * (1.0 + 0.10 * i / (n - 1))
        price_b = 200.0 * (1.0 + 0.10 * i / (n - 1))
        prices[("ASSET_A", date_str)] = price_a
        prices[("ASSET_B", date_str)] = price_b

    return prices


PRICE_MAP = _build_price_series()


def _make_mock_data():
    """Create a MagicMock DataFetcher wired to synthetic data."""
    data = MagicMock()

    known = {"ASSET_A": "Asset A", "ASSET_B": "Asset B"}

    def fetch_reference_data(identifier,
                             date_str):
        # One name or a list of them, as the real fetcher takes (BN-192).
        wanted = [name for name
                  in ([identifier] if isinstance(identifier, str) else identifier)
                  if name in known]

        if not wanted:
            return pd.DataFrame()

        return pd.DataFrame(
            {"NAME": [known[name] for name in wanted],
             "CURRENCY": ["USD"] * len(wanted),
             "EXCHANGE": ["NYSE"] * len(wanted)},
            index=pd.Index(wanted, name="IDENTIFIER"),
        )

    def fetch_market_data(ticker,
                          start,
                          end):
        # Return a single-row DataFrame for the requested date
        key = (ticker, start)
        if key in PRICE_MAP:
            price = PRICE_MAP[key]
            return pd.DataFrame(
                {"CLOSE": [price]},
                index=pd.Index([pd.Timestamp(start)], name="DATE"),
            )
        return pd.DataFrame()

    def fetch_shares_outstanding(ticker,
                                 date_str):
        return SHARES

    data.fetch_reference_data.side_effect = fetch_reference_data
    data.fetch_market_data.side_effect = fetch_market_data
    data.fetch_shares_outstanding.side_effect = fetch_shares_outstanding
    return data


@pytest.fixture
def index_definition():
    return IndexDefinition(
        index_id="TEST_EW",
        index_name="Test Equal Weight",
        base_date=BASE_DATE,
        base_value=BASE_VALUE,
        currency="USD",
        eligibility_rules=[],  # pass-all
        weighting_scheme=EqualWeighted(),
        rebalancing_frequency="MONTHLY",
        calendar="XNYS",
        universe_identifiers=["ASSET_A", "ASSET_B"],
    )


@pytest.fixture
def mock_data():
    return _make_mock_data()


@pytest.fixture
def result(index_definition,
           mock_data):
    """Run the calculator once and return the IndexResult."""
    calc = IndexCalculator(index_definition, mock_data)
    return calc.run(end_date=END_DATE)


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestIntegrationRun:

    def test_returns_index_result(self,
                                  result):
        assert isinstance(result, IndexResult)
        assert result.index_id == "TEST_EW"

    def test_base_date_level_equals_base_value(self,
                                               result):
        base = pd.Timestamp(BASE_DATE)
        assert result.index_levels[base] == pytest.approx(BASE_VALUE)

    def test_level_continuous_across_rebalances(self,
                                                result):
        """No jumps at rebalance dates — the level the day before and
        the rebalance date itself should be close (within daily move)."""
        levels = result.index_levels
        rebalance_dates = set(result.weight_snapshots.keys())
        rebalance_dates.discard(pd.Timestamp(BASE_DATE))

        for rdate in rebalance_dates:
            idx = levels.index.get_loc(rdate)
            if idx == 0:
                continue
            prev_level = levels.iloc[idx - 1]
            rebal_level = levels.iloc[idx]
            # With equal weights and same constituents the jump should be
            # purely from daily price change, not from divisor discontinuity.
            # Daily move is at most ~0.5% for our linear 10%-over-3-months data.
            pct_change = abs(rebal_level - prev_level) / prev_level
            assert pct_change < 0.01, (
                f"Level jump of {pct_change:.4%} at rebalance {rdate} exceeds 1%"
            )

    def test_equal_weight_10pct_gain(self,
                                     result):
        """Both assets gain 10%. Equal-weight index should also gain ~10%."""
        final_level = result.index_levels.iloc[-1]
        total_return = (final_level / BASE_VALUE) - 1.0
        assert total_return == pytest.approx(0.10, abs=0.005)

    def test_divisor_changes_only_on_rebalance_dates(self,
                                                     result):
        """Divisor should only change on dates recorded in weight_snapshots."""
        divisors = result.divisor_history
        rebalance_dates = set(result.weight_snapshots.keys())

        for i in range(1, len(divisors)):
            date = divisors.index[i]
            if date not in rebalance_dates:
                assert divisors.iloc[i] == pytest.approx(divisors.iloc[i - 1]), (
                    f"Divisor changed on non-rebalance date {date}"
                )

    def test_weight_snapshots_only_on_rebalance_dates(self,
                                                      result):
        """Weight entries should correspond to base + rebalance dates."""
        rebal_dates_from_defn = {
            pd.Timestamp(d) for d in result.weight_snapshots
        }
        # All snapshot dates should be in the trading range
        trading_days = set(result.index_levels.index)
        assert rebal_dates_from_defn.issubset(trading_days)

        # Base date must be present
        assert pd.Timestamp(BASE_DATE) in rebal_dates_from_defn

    def test_weights_are_equal(self,
                               result):
        """All weight snapshots should show 50/50 split."""
        for weights in result.weight_snapshots.values():
            assert len(weights) == 2
            for w in weights.values():
                assert w == pytest.approx(0.5)

    def test_return_calculation_matches_manual(self,
                                               result):
        """Spot-check: first daily return should match manual computation."""
        levels = result.index_levels
        day0 = levels.index[0]
        day1 = levels.index[1]
        expected_return = (levels[day1] - levels[day0]) / levels[day0]

        returns = result.get_returns()
        assert returns.iloc[0] == pytest.approx(expected_return)

    def test_all_returns_positive(self,
                                  result):
        """Both assets only go up, so every daily return should be >= 0."""
        returns = result.get_returns()
        assert (returns >= -1e-10).all()

    def test_idempotent(self,
                        index_definition):
        """Two consecutive run() calls produce identical results."""
        data1 = _make_mock_data()
        data2 = _make_mock_data()
        calc1 = IndexCalculator(index_definition, data1)
        calc2 = IndexCalculator(index_definition, data2)

        r1 = calc1.run(end_date=END_DATE)
        r2 = calc2.run(end_date=END_DATE)

        pd.testing.assert_series_equal(r1.index_levels, r2.index_levels)
        pd.testing.assert_series_equal(r1.divisor_history, r2.divisor_history)
        assert r1.constituent_snapshots == r2.constituent_snapshots
        assert r1.weight_snapshots == r2.weight_snapshots

    def test_covers_expected_trading_days(self,
                                          result):
        """Every session from base to end, and only those.

        Business days until BN-186, which over this span counted three days
        NYSE was shut: Martin Luther King Day (2024-01-15), Presidents' Day
        (2024-02-19) and Good Friday (2024-03-29). The data carries a bar on
        each of them — the fixture is built from `bdate_range` — so before the
        fix the index published a level on all three. 64 business days, 61
        sessions.
        """
        expected_days = sessions(pd.Timestamp(BASE_DATE),
                                 pd.Timestamp(END_DATE), CALENDAR)

        assert len(result.index_levels) == len(expected_days)
        assert list(result.index_levels.index) == list(expected_days)

    def test_constituent_snapshots_contain_both_assets(self,
                                                       result):
        for ids in result.constituent_snapshots.values():
            assert set(ids) == {"ASSET_A", "ASSET_B"}


class TestACalendarThatCannotCoverTheRange:
    """BN-198. A window the calendar cannot reach used to succeed, emptily.

    `sessions()` clamps to what the calendar object holds, so a pre-1997 Tokyo
    index got no sessions at all, and `run()` logged "no trading days" and
    returned an `IndexResult` with no levels in it. Good data, a real request,
    a successful result containing nothing, and a WARNING as the only record
    that anything had gone wrong.

    Karan's call on the two halves: refuse when the calendar covers none of the
    window, because an empty series is a failure wearing a success and neither
    half of the remedy is discoverable from an empty list; report when it
    covers part, because that is a real answer to a smaller question.
    """

    def definition(self,
                   base_date: str,
                   calendar: str) -> IndexDefinition:
        return IndexDefinition(
            index_id="TZ",
            index_name="Calendar Coverage",
            base_date=base_date,
            base_value=BASE_VALUE,
            # Matching the fixture's reference data, so no FX conversion is
            # attempted: the universe is quoted in USD and a JPY index would
            # reach for a rate the mock provider answers with a MagicMock.
            currency="USD",
            eligibility_rules=[],
            weighting_scheme=EqualWeighted(),
            rebalancing_frequency="QUARTERLY",
            calendar=calendar,
            universe_identifiers=["ASSET_A", "ASSET_B"],
        )

    def test_it_refuses_rather_than_publishing_an_empty_index(self,
                                                              mock_data):
        calculator = IndexCalculator(self.definition("1995-01-02", "XTKS"),
                                     mock_data)

        with pytest.raises(CalculationError):
            calculator.run(end_date="1995-12-29")

    def test_the_refusal_names_the_calendar_and_its_earliest_session(self,
                                                                     mock_data):
        """"The calendar does not cover 1995" leaves a reader looking at the
        base date, which is not the thing that is wrong."""
        calculator = IndexCalculator(self.definition("1995-01-02", "XTKS"),
                                     mock_data)

        with pytest.raises(CalculationError) as raised:
            calculator.run(end_date="1995-12-29")

        message = str(raised.value)

        assert "XTKS" in message
        assert "Tokyo Stock Exchange" in message
        assert "1997-01-06" in message

    def test_an_ordinary_run_reports_no_coverage_at_all(self,
                                                        result):
        """Null on a full cover, so its presence is itself the signal."""
        assert result.calendar_coverage is None

    def priced_over(self,
                    start: str,
                    end: str) -> MagicMock:
        """A provider carrying flat prices across an arbitrary period.

        The module fixture only holds 2024, and a 1997 run over it refuses on
        an unpriced constituent (BN-191) long before it can say anything about
        calendars — a different refusal reached first, which is the shape
        BN-196 was about.
        """
        data = _make_mock_data()
        priced = set(pd.bdate_range(start, end).strftime("%Y-%m-%d"))

        def fetch_market_data(ticker,
                              begin,
                              finish):
            if str(begin)[:10] not in priced:
                return pd.DataFrame()

            return pd.DataFrame(
                {"CLOSE": [100.0]},
                index=pd.Index([pd.Timestamp(begin)], name="DATE"))

        data.fetch_market_data.side_effect = fetch_market_data

        return data

    def test_a_partial_cover_succeeds_and_carries_both_ranges(self):
        """XTKS reaches 1997, so a 1996 start is narrowed rather than refused."""
        calculator = IndexCalculator(self.definition("1996-01-02", "XTKS"),
                                     self.priced_over("1996-01-01", "1999-01-31"))

        coverage = calculator.run(end_date="1998-12-31").calendar_coverage

        assert coverage is not None
        assert coverage.requested_start == pd.Timestamp("1996-01-02")
        assert coverage.covered_start == pd.Timestamp("1997-01-06")
        assert coverage.trimmed_start
        assert not coverage.trimmed_end

    def test_the_levels_stop_at_the_covered_range(self):
        """The report is only worth having if it describes what was produced."""
        calculator = IndexCalculator(self.definition("1996-01-02", "XTKS"),
                                     self.priced_over("1996-01-01", "1999-01-31"))

        result = calculator.run(end_date="1998-12-31")

        assert result.index_levels.index.min() >= pd.Timestamp("1997-01-06")
