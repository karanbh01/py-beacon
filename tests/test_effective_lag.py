# tests/test_effective_lag.py
"""BN-126: a composition announced on one date and applied on another.

Real indices publish a constituent list before it takes effect, which is what
gives tracking funds time to trade. Until now a rebalance was one instant.

The load-bearing test is `TestZeroLagIsUnchanged`: an index without a lag must
produce byte-identical levels to one calculated before this existed. Everything
else here adds behaviour; that proves the addition took nothing away.

Snapshots are keyed by the **effective** date, because that is when the weights
are in force — which is what drift, attribution and the backtest engine all
need. The announcement travels alongside rather than instead, since a client
showing "rebalance of 3 Apr, effective 6 Apr" needs both.
"""
import pandas as pd
import pytest

from beacon.backtest.engine import BacktestEngine
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import MarketCapWeighted
from beacon.index.schedule import effective_date, sessions
from beacon.testing import dataset

START = "2023-01-02"
END = "2024-06-28"

# The day the index is actually based on. START is a Monday and the day NYSE
# observed New Year's Day, so it is not a session; BN-186 rolls the base date
# forward onto the first one, which is also the first scheduled rebalance.
BASE_SESSION = pd.Timestamp("2023-01-03")


def build(lag: int = 0) -> IndexDefinition:
    """A market-cap index over the canonical universe, with a lag."""
    return IndexDefinition(
        index_id="IDX", index_name="Index", base_date=START, base_value=1000.0,
        currency="USD", eligibility_rules=[],
        weighting_scheme=MarketCapWeighted(),
        rebalancing_frequency="QUARTERLY",
        calendar="XNYS",
        universe_identifiers=list(dataset.UNIVERSE),
        effective_lag_sessions=lag)


def run(lag: int = 0):
    """Calculate the index."""
    return IndexCalculator(build(lag), dataset.data_fetcher()).run(
        start_date=START, end_date=END)


@pytest.fixture(scope="module")
def unlagged():
    return run(0)


@pytest.fixture(scope="module")
def lagged():
    return run(3)


class TestZeroLagIsUnchanged:
    """The addition must take nothing away."""

    def test_the_levels_are_identical(self, unlagged):
        """Against an index built with no lag argument at all, which is how
        every definition predating this issue is constructed."""
        without_the_field = IndexCalculator(
            IndexDefinition(
                index_id="IDX", index_name="Index", base_date=START,
                base_value=1000.0, currency="USD", eligibility_rules=[],
                weighting_scheme=MarketCapWeighted(),
                rebalancing_frequency="QUARTERLY",
                calendar="XNYS",
                universe_identifiers=list(dataset.UNIVERSE)),
            dataset.data_fetcher()).run(start_date=START, end_date=END)

        pd.testing.assert_series_equal(unlagged.index_levels,
                                       without_the_field.index_levels)

    def test_no_announcement_map_is_carried(self, unlagged):
        """Empty rather than an identity mapping, so its presence is itself the
        signal that a lag applies."""
        assert unlagged.announcement_dates == {}

    def test_the_rebalance_dates_are_the_schedule(self, unlagged):
        """Exactly the schedule, with nothing extra alongside it.

        BN-180 left this as "the schedule *plus* the base date": 2023-01-02 is
        a Monday and the day NYSE observed New Year's Day, so the schedule's
        first January date was the 3rd while the index still initialised on
        the 2nd — a snapshot on a day the market was shut, because the loop
        counted business days. BN-186 rolls the base date onto the first
        session, which is the 3rd, so the two coincide and the extra snapshot
        is gone.
        """
        dates = sorted(unlagged.weight_snapshots)
        scheduled = build(0).get_rebalance_dates(START, END)

        assert dates == sorted(scheduled)
        assert dates[0] == BASE_SESSION


class TestLaggedApplication:
    """Announced on one date, in force on another."""

    def test_every_rebalance_records_its_announcement(self, lagged):
        rebalances = [date for date in sorted(lagged.weight_snapshots)
                      if date != BASE_SESSION]

        assert len(lagged.announcement_dates) == len(rebalances)

    def test_the_announcement_precedes_the_effective_date(self, lagged):
        for effective, announced in lagged.announcement_dates.items():
            assert announced < effective

    def test_the_gap_is_the_configured_number_of_sessions(self, lagged):
        """Counted in the index's own sessions, not in business days. The two
        gave the same answer until BN-180, when the definition gained a real
        calendar: a July rebalance announced on the 3rd is effective on the 7th
        because 4 July is not a session, and a business-day ruler reads that as
        four days rather than three.
        """
        for effective, announced in lagged.announcement_dates.items():
            between = sessions(announced, effective, build(3).calendar)

            assert len(between) - 1 == 3

    def test_the_announcements_are_the_unlagged_schedule(self, lagged):
        """The lag moves when a composition is applied, not when it is
        decided — so the announcements are the ordinary schedule."""
        announced = sorted(lagged.announcement_dates.values())
        scheduled = [date for date in build(0).get_rebalance_dates(START, END)
                     if date != BASE_SESSION]

        assert announced == scheduled

    def test_snapshots_are_keyed_by_the_effective_date(self, lagged, unlagged):
        lagged_dates = set(sorted(lagged.weight_snapshots)[1:])
        unlagged_dates = set(sorted(unlagged.weight_snapshots)[1:])

        assert lagged_dates.isdisjoint(unlagged_dates)
        assert lagged_dates == set(lagged.announcement_dates)

    def test_the_levels_differ_from_the_unlagged_index(self, lagged, unlagged):
        """Applying the same weights days later is a different index, and one
        that produced identical levels would mean the lag did nothing."""
        assert not lagged.index_levels.equals(unlagged.index_levels)

    def test_the_weights_do_not_change_on_the_announcement(self, lagged):
        """The whole point. On the announcement date the index still holds what
        it held the day before."""
        for effective, announced in lagged.announcement_dates.items():
            assert announced not in lagged.weight_snapshots
            assert effective in lagged.weight_snapshots


class TestSelectionIsAsOfTheAnnouncement:
    """What was published is what gets applied."""

    def test_the_composition_matches_an_unlagged_run_at_the_announcement(self):
        """Selection and weighting happen as of the announcement, even though
        the units are built at the effective date's prices. Selecting on the
        effective date instead would let a price move in between drop a name
        that qualified when the index was announced."""
        lagged = run(3)
        unlagged = run(0)

        for effective, announced in lagged.announcement_dates.items():
            assert (set(lagged.weight_snapshots[effective])
                    == set(unlagged.weight_snapshots[announced])), announced


class TestEffectiveDateHelper:
    """The date arithmetic on its own."""

    @pytest.fixture
    def panel(self) -> pd.DatetimeIndex:
        return pd.bdate_range("2025-01-01", "2025-03-31")

    def test_zero_lag_is_the_announcement(self, panel):
        announced = pd.Timestamp("2025-01-15")

        assert effective_date(announced, 0, panel) == announced

    def test_a_negative_lag_is_the_announcement(self, panel):
        announced = pd.Timestamp("2025-01-15")

        assert effective_date(announced, -2, panel) == announced

    def test_it_counts_sessions_not_calendar_days(self, panel):
        """Friday plus two sessions is Tuesday, not Sunday."""
        friday = pd.Timestamp("2025-01-17")

        assert effective_date(friday, 2, panel) == pd.Timestamp("2025-01-21")

    def test_a_lag_past_the_end_of_the_panel_falls_back(self, panel, caplog):
        """An index whose data ends mid-lag should apply its last rebalance
        rather than drop it."""
        import logging

        near_the_end = pd.Timestamp("2025-03-28")

        with caplog.at_level(logging.WARNING):
            result = effective_date(near_the_end, 10, panel)

        assert result == near_the_end
        assert "fewer than" in caplog.text


class TestServedSnapshots:
    """What a client sees."""

    def _snapshot_payloads(self, result):
        from beacon.server.backtests import rebalance_snapshots

        return rebalance_snapshots(result, cap=None)

    def test_an_unlagged_snapshot_announces_nothing(self, unlagged):
        """Null rather than repeating the date, so a client can tell a lagged
        index from one that simply announces and applies together."""
        payloads = self._snapshot_payloads(unlagged)

        assert all(entry.announced is None for entry in payloads)

    def test_a_lagged_snapshot_carries_both_dates(self, lagged):
        payloads = self._snapshot_payloads(lagged)
        with_announcements = [entry for entry in payloads if entry.announced]

        assert with_announcements
        for entry in with_announcements:
            assert entry.announced < entry.date


class TestBacktestTradesOnEffectiveDates:
    """The engine follows the weight schedule, which is keyed by effective."""

    def test_it_trades_on_effective_dates_only(self, lagged):
        """No engine change was needed: keying snapshots by the effective date
        is what makes the engine trade on them. This asserts that, because it
        is the kind of thing that holds by accident until it does not."""
        result = BacktestEngine(start_date=START, end_date=END,
                                initial_capital=1_000_000.0,
                                data_provider=dataset.data_fetcher(),
                                index_result=lagged,
                                transaction_cost_bps=0.0).run()

        traded = {pd.Timestamp(transaction.transaction_date)
                  for transaction in result.portfolio.transactions}

        announcements = set(lagged.announcement_dates.values())
        effectives = set(lagged.announcement_dates)

        assert traded & effectives, "no trades on any effective date"
        assert not traded & announcements, (
            "traded on an announcement date, before the weights were in force")
