# tests/test_data_returns.py
"""BN-362: each name's daily total return, in one currency.

Three names on fixed prices over a month: one in dollars with a split and a
dividend, one in pounds that pays a dividend while the pound moves, and one
with a gap in its prices. Each return is checked against the definition,
worked by hand.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.data import pair_reference
from beacon.data.base import MarketData, ReferenceData
from beacon.data.corporate_actions import CorporateActions
from beacon.data.fetcher import DataFetcher
from beacon.data.returns import name_returns

DAYS = pd.bdate_range("2024-03-01", periods=20)
SPLIT_DAY = DAYS[5]
DIVIDEND_DAY = DAYS[10]
GAP_DAY = DAYS[7]

# AAA climbs 1% a day in dollars, halving in price on its 2-for-1 split.
AAA = [100.0 * 1.01 ** step * (0.5 if day >= SPLIT_DAY else 1.0)
       for step, day in enumerate(DAYS)]
# BBB is flat at 200 pence until its dividend of 4, then 196, while the
# pound climbs from 1.25 dollars by a tenth of a cent a day.
BBB = [200.0 if day < DIVIDEND_DAY else 196.0 for day in DAYS]
GBPUSD = [1.25 + 0.001 * step for step in range(len(DAYS))]
# CCC climbs 2 dollars a day but does not print on one day.
CCC = [50.0 + 2.0 * step for step in range(len(DAYS))]


def fetcher(actions: list[dict] | None = None) -> DataFetcher:
    rows = []
    for name, closes in (("AAA", AAA), ("BBB", BBB), ("CCC", CCC), ("GBPUSD", GBPUSD)):
        rows += [{"IDENTIFIER": name, "DATE": day, "CLOSE": close}
                 for day, close in zip(DAYS, closes, strict=True)
                 if not (name == "CCC" and day == GAP_DAY)]

    reference = pd.DataFrame([
        {"IDENTIFIER": "AAA", "DATE_FROM": "2020-01-01", "NAME": "A", "CURRENCY": "USD",
         "EXCHANGE": "XNYS"},
        {"IDENTIFIER": "BBB", "DATE_FROM": "2020-01-01", "NAME": "B", "CURRENCY": "GBP",
         "EXCHANGE": "XLON"},
        {"IDENTIFIER": "CCC", "DATE_FROM": "2020-01-01", "NAME": "C", "CURRENCY": "USD",
         "EXCHANGE": "XNYS"},
        pair_reference("GBPUSD")])

    if actions is None:
        actions = [{"IDENTIFIER": "AAA", "EX_DATE": SPLIT_DAY, "TYPE": "SPLIT", "VALUE": 2.0},
                   {"IDENTIFIER": "AAA", "EX_DATE": DAYS[15], "TYPE": "DIVIDEND", "VALUE": 0.5},
                   {"IDENTIFIER": "BBB", "EX_DATE": DIVIDEND_DAY, "TYPE": "DIVIDEND",
                    "VALUE": 4.0}]

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference),
                       CorporateActions.from_dataframe(pd.DataFrame(actions)))


def returns(data: DataFetcher | None = None,
            currency: str | None = "USD") -> pd.DataFrame:
    return name_returns(data or fetcher(), ["AAA", "BBB", "CCC"],
                        str(DAYS[0].date()), str(DAYS[-1].date()), currency)


def at(frame: pd.DataFrame,
       day: pd.Timestamp,
       name: str) -> float:
    return float(frame.loc[day, name])


class TestTheDefinition:

    def test_an_ordinary_day_is_the_close_over_the_last(self):
        assert at(returns(), DAYS[3], "AAA") == pytest.approx(0.01, rel=1e-12)

    def test_the_first_day_has_no_close_before_it(self):
        assert returns().loc[DAYS[0]].isna().all()

    def test_a_split_is_not_a_fall(self):
        assert at(returns(), SPLIT_DAY, "AAA") == pytest.approx(0.01, rel=1e-12)

    def test_a_dividend_counts_on_its_ex_date(self):
        day = DAYS[15]
        expected = (AAA[15] + 0.5) / AAA[14] - 1.0

        assert at(returns(), day, "AAA") == pytest.approx(expected, rel=1e-12)


class TestCurrency:

    def test_a_name_s_own_currency_when_none_is_asked(self):
        own = returns(currency=None)

        assert at(own, DAYS[3], "BBB") == pytest.approx(0.0, abs=1e-15)
        assert at(own, DIVIDEND_DAY, "BBB") == pytest.approx(0.0, abs=1e-15)

    def test_the_rate_s_move_is_part_of_the_return(self):
        expected = GBPUSD[3] / GBPUSD[2] - 1.0

        assert at(returns(), DAYS[3], "BBB") == pytest.approx(expected, rel=1e-12)

    def test_cash_is_converted_at_the_ex_date_s_rate(self):
        position = DAYS.get_loc(DIVIDEND_DAY)
        expected = ((196.0 + 4.0) * GBPUSD[position]) / (200.0 * GBPUSD[position - 1]) - 1.0

        assert at(returns(), DIVIDEND_DAY, "BBB") == pytest.approx(expected, rel=1e-12)


class TestGaps:

    def test_a_day_without_a_close_is_empty(self):
        assert np.isnan(at(returns(), GAP_DAY, "CCC"))

    def test_the_next_close_runs_from_the_last_one(self):
        position = DAYS.get_loc(GAP_DAY)
        expected = CCC[position + 1] / CCC[position - 1] - 1.0

        assert at(returns(), DAYS[position + 1], "CCC") == pytest.approx(expected, rel=1e-12)


class TestWhichActionsCount:

    def test_a_cancelled_dividend_is_left_out(self):
        data = fetcher([{"IDENTIFIER": "BBB", "EX_DATE": DIVIDEND_DAY, "TYPE": "DIVIDEND",
                         "VALUE": 4.0, "STATUS": "CANCELLED"}])
        position = DAYS.get_loc(DIVIDEND_DAY)
        expected = (196.0 * GBPUSD[position]) / (200.0 * GBPUSD[position - 1]) - 1.0

        assert at(returns(data), DIVIDEND_DAY, "BBB") == pytest.approx(expected, rel=1e-12)

    def test_a_merger_is_not_a_return(self):
        data = fetcher([{"IDENTIFIER": "AAA", "EX_DATE": DAYS[3], "TYPE": "MERGER",
                         "VALUE": 3.0}])

        assert at(returns(data), DAYS[3], "AAA") == pytest.approx(0.01, rel=1e-12)

    def test_an_action_between_sessions_counts_on_the_next(self):
        saturday = DAYS[5] + pd.Timedelta(days=1)
        assert DAYS[6] - DAYS[5] == pd.Timedelta(days=3)
        data = fetcher([{"IDENTIFIER": "CCC", "EX_DATE": saturday, "TYPE": "DIVIDEND",
                         "VALUE": 1.0}])
        expected = (CCC[6] + 1.0) / CCC[5] - 1.0

        assert at(returns(data), DAYS[6], "CCC") == pytest.approx(expected, rel=1e-12)

    def test_without_any_actions_returns_are_the_closes(self):
        data = fetcher([{"IDENTIFIER": "ZZZ", "EX_DATE": DAYS[3], "TYPE": "DIVIDEND",
                         "VALUE": 1.0}])
        frame = returns(data, currency=None)

        assert at(frame, SPLIT_DAY, "AAA") == pytest.approx(AAA[5] / AAA[4] - 1.0, rel=1e-12)


class TestShape:

    def test_dates_by_name(self):
        frame = returns()

        assert list(frame.columns) == ["AAA", "BBB", "CCC"]
        assert frame.index.equals(DAYS)

    def test_a_name_without_prices_is_left_out(self):
        frame = name_returns(fetcher(), ["AAA", "NOPE"], str(DAYS[0].date()),
                             str(DAYS[-1].date()), "USD")

        assert list(frame.columns) == ["AAA"]
