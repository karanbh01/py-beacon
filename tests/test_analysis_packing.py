# tests/test_analysis_packing.py
"""BN-403: a run's closes kept packed, and the market inputs that stand in
for the data store when its returns are worked out again.

Closes at whole cents, at four decimals and at full precision, with gaps;
and the three-name market of the returns tests, with a split, dividends and
a pound-quoted name.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.packing import (
    CLOSES_FLOAT,
    CLOSES_PACKED,
    CLOSES_SCALE,
    StoredSource,
    decimals_of,
    market_inputs,
    pack_closes,
    unpack_closes,
)
from beacon.data.returns import name_returns

DAYS = pd.bdate_range("2024-01-02", periods=30)


def closes() -> pd.DataFrame:
    random = np.random.default_rng(11)
    cents = np.round(50 + random.random(len(DAYS)) * 10, 2)
    fine = np.round(0.5 + random.random(len(DAYS)), 4)
    noisy = 100 + random.random(len(DAYS))
    frame = pd.DataFrame({"CENTS": cents, "NOISY": noisy, "FINE": fine}, index=DAYS)
    frame.iloc[3:6, 0] = np.nan

    return frame


class TestPacking:

    def test_the_fewest_decimals_that_hold_every_value(self):
        assert decimals_of(np.array([1.0, 2.0, np.nan])) == 0
        assert decimals_of(np.array([1.25, 99.99])) == 2
        assert decimals_of(np.array([0.1234])) == 4
        assert decimals_of(np.array([np.pi])) is None

    def test_closes_come_back_exactly(self):
        frame = closes()

        back = unpack_closes(pack_closes(frame), list(frame.columns))

        np.testing.assert_array_equal(back.to_numpy(), frame.to_numpy())
        assert list(back.columns) == list(frame.columns)
        assert back.index.equals(frame.index)

    def test_each_name_is_kept_at_its_own_scale(self):
        tables = pack_closes(closes())

        assert tables[CLOSES_SCALE]["decimals"].to_dict() == {"CENTS": 2, "FINE": 4}
        assert list(tables[CLOSES_FLOAT].columns) == ["NOISY"]
        assert str(tables[CLOSES_PACKED]["CENTS"].dtype) == "Int64"

    def test_a_packed_table_reads_back_exactly_from_disk(self,
                                                         tmp_path):
        from beacon.analysis.panels import read_panels, write_panels

        frame = closes()
        write_panels(tmp_path / "run", pack_closes(frame))

        back = unpack_closes(read_panels(tmp_path / "run"), list(frame.columns))

        np.testing.assert_array_equal(back.to_numpy(), frame.to_numpy())


class TestStoredSource:
    """The returns worked out from the kept inputs are the data store's."""

    @pytest.fixture(scope="class")
    def market(self):
        from beacon.data.prices import prices_in
        from test_data_returns import DAYS as RETURN_DAYS
        from test_data_returns import fetcher

        data = fetcher()
        local = prices_in(data, ["AAA", "BBB", "CCC"], str(RETURN_DAYS[0].date()),
                          str(RETURN_DAYS[-1].date()))

        return data, local

    def test_returns_from_the_kept_inputs_equal_the_store_s(self,
                                                          market):
        data, local = market
        inputs = market_inputs(data, local, "USD")
        stored = StoredSource(inputs["currencies"], inputs["rates"], inputs["actions"], "USD")

        kept = name_returns(stored, list(local.columns), currency="USD", closes=local)
        live = name_returns(data, list(local.columns), currency="USD", closes=local)

        assert kept.equals(live)
        assert np.isfinite(kept.to_numpy()).sum() > 50

    def test_only_currencies_away_from_the_book_have_rates(self,
                                                          market):
        data, local = market
        inputs = market_inputs(data, local, "USD")

        assert inputs["currencies"]["CURRENCY"].to_dict() == {"AAA": "USD", "BBB": "GBP",
                                                              "CCC": "USD"}
        assert list(inputs["rates"].columns) == ["GBP"]

    def test_the_actions_kept_are_the_names_own(self,
                                                market):
        data, local = market
        inputs = market_inputs(data, local[["BBB"]], "USD")

        assert set(inputs["actions"]["IDENTIFIER"]) == {"BBB"}

    def test_a_currency_without_a_kept_rate_is_refused_as_the_store_refuses_it(self,
                                                                             market):
        from beacon.exceptions import CalculationError

        data, local = market
        inputs = market_inputs(data, local, "USD")
        stored = StoredSource(inputs["currencies"], inputs["rates"].drop(columns="GBP"),
                              inputs["actions"], "USD")

        with pytest.raises(CalculationError, match="GBP"):
            name_returns(stored, list(local.columns), currency="USD", closes=local)
