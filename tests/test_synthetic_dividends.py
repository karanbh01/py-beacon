# tests/test_synthetic_dividends.py
"""BN-357: each generated name has its own dividend timing, so a universe's
dividends land across the year rather than on one day a quarter."""
import pandas as pd
import pytest

from beacon.index.schedule import sessions
from beacon.synthetic import dividends

CALENDAR = sessions(pd.Timestamp("2023-01-03"), pd.Timestamp("2024-12-31"), "XNYS")
NAMES = pd.Index([f"CMP{i:03d}" for i in range(300)])


class TestTimings:

    def test_a_name_keeps_its_timing_alongside_any_others(self):
        alone = dividends.timings(NAMES[:1], 7, dividends.SPREAD)
        together = dividends.timings(NAMES, 7, dividends.SPREAD)

        assert alone.loc[NAMES[0]].tolist() == together.loc[NAMES[0]].tolist()

    def test_another_seed_times_it_differently(self):
        first = dividends.timings(NAMES, 7, dividends.SPREAD)
        second = dividends.timings(NAMES, 8, dividends.SPREAD)

        assert not first.equals(second)

    def test_spread_covers_every_cycle_day_and_settlement(self):
        timing = dividends.timings(NAMES, 7, dividends.SPREAD)

        assert set(timing["cycle"]) == {0, 1, 2}
        assert timing["day"].between(1, dividends.LAST_DAY).all()
        assert timing["day"].nunique() > 20
        assert timing["settlement"].between(*dividends.SETTLEMENT_DAYS).all()

    def test_fixed_is_the_old_timing(self):
        timing = dividends.timings(NAMES[:3], 7, dividends.FIXED)

        assert timing.to_numpy().tolist() == [[1, 15, 21]] * 3

    def test_an_unknown_schedule_is_refused(self):
        with pytest.raises(ValueError, match="Unknown dividend schedule"):
            dividends.timings(NAMES, 7, "monthly")


class TestExDates:

    def test_once_a_quarter_in_its_cycle(self):
        dates = dividends.ex_dates(CALENDAR, 2, 10)

        assert len(dates) == 8
        assert sorted(set(dates.month)) == [3, 6, 9, 12]
        assert (dates.day >= 10).all()

    def test_a_late_day_falls_back_to_the_months_last_session(self):
        """No session on or after the 31st in June: its last one stands in,
        so no quarter is skipped."""
        dates = dividends.ex_dates(CALENDAR, 2, 31)

        assert len(dates) == 8
        assert pd.Timestamp("2023-06-30") in dates

    def test_not_in_a_month_the_calendar_stops_part_way_through(self):
        cutoff = pd.Timestamp("2024-12-10")
        short = CALENDAR[CALENDAR.to_series().le(cutoff).to_numpy()]

        assert dividends.ex_dates(short, 2, 20)[-1] == pd.Timestamp("2024-09-20")
