# tests/test_library_from_server.py
"""BN-417: the smaller figures the server used to work out itself, now the
library's: market caps and their shares, a book's held weights, the cash
buffer, a cap's reach, the rebalance schedule's horizon, and a parameter's
value in words."""
import pandas as pd
import pytest

from beacon.analysis.concentration import total_weight
from beacon.analysis.liquidity import trailing_start
from beacon.backtest.implementation import buffered
from beacon.catalogue import BPS, DAYS, FRACTION, MONEY, in_words
from beacon.data.market_caps import caps_of, is_stale, shares_of_total
from beacon.index.capping import is_tight, reachable_weight
from beacon.index.schedule import days_until, projection_end
from beacon.portfolio.valuation import held_weights


class TestMarketCaps:

    def test_each_cap_from_its_own_base(self):
        caps = caps_of(10.0, 1e6, 1.25, 0.8)

        assert (caps.local, caps.converted) == (1e7, 1.25e7)
        assert caps.free_float_local == pytest.approx(8e6)
        assert caps.free_float_converted == pytest.approx(1e7)

    def test_unknowns_stay_unknown(self):
        caps = caps_of(10.0, 1e6, None, None)

        assert caps.converted is None
        assert caps.free_float_local is None and caps.free_float_converted is None

    def test_staleness(self):
        day = pd.Timestamp("2024-03-01")

        assert is_stale(day - pd.Timedelta(days=31), day, 30)
        assert not is_stale(day - pd.Timedelta(days=30), day, 30)

    def test_shares_of_the_positive_total(self):
        shares = shares_of_total({"A": 3.0, "B": 1.0, "C": None, "D": -2.0})

        assert shares == {"A": 0.75, "B": 0.25, "C": None, "D": -0.5}
        assert shares_of_total({"A": None}) == {"A": None}


class TestBook:

    def test_held_weights(self):
        prices = pd.Series({"A": 10.0, "B": float("nan")})

        assert held_weights({"A": 50.0, "B": 10.0, "C": 1.0}, prices, 1000.0) == {
            "A": (50.0, 0.5)}

    def test_the_cash_buffer_scales_every_weight(self):
        assert buffered({"A": 0.6, "B": 0.4}, 0.05) == {"A": pytest.approx(0.57),
                                                         "B": pytest.approx(0.38)}

    def test_total_weight(self):
        assert total_weight({"A": 0.25, "B": 0.5}) == 0.75


class TestCaps:

    def test_reach_and_tightness(self):
        assert reachable_weight(0.1, 9) == pytest.approx(0.9)
        assert is_tight(0.1, 10)
        assert not is_tight(0.1, 12)


class TestDates:

    def test_the_schedule_horizon(self):
        day = pd.Timestamp("2024-01-15")

        assert projection_end("QUARTERLY", day, 4) == pd.Timestamp("2025-01-15")
        assert days_until(pd.Timestamp("2024-02-14"), day) == 30

    def test_the_trailing_window(self):
        assert trailing_start(pd.Timestamp("2024-05-31")) == pd.Timestamp("2024-02-29")


class TestInWords:

    @pytest.mark.parametrize(("value", "unit", "words"),
                             [(0.05, FRACTION, "5%"), (12.5, BPS, "12.5 bps"),
                              (1.0, DAYS, "1 day"), (3.0, DAYS, "3 days"),
                              (1500.0, MONEY, "1,500 USD"), (2.5, None, "2.5")])
    def test_a_value_with_its_unit(self,
                                   value,
                                   unit,
                                   words):
        assert in_words(value, unit, "USD") == words
