# tests/test_backtest_pricing.py
"""BN-183: the engine prices from a session, and says when it had to.

The bug these were written against: `_fetch_price` read one exact date, so a
rebalance scheduled on a day the market was shut found no price, both
instruction builders returned None, and the run recorded a rebalance on which
nothing moved. Attribution and the weights history read those snapshots
afterwards, so a record that trades nothing is not a cosmetic fault.

Three absences look identical to an exact-date read and mean different things,
which is what the tests here separate:

* **A market holiday.** The calendar says the day was closed, so the previous
  session's price is what the position was worth through it. Resolved back
  silently — nothing is missing.
* **A hole in the data.** The calendar says the day was open, so a bar should
  exist. The price is carried forward and the fact is published, rather than a
  five-hundred-name backtest dying over one bad day.
* **A name that stopped existing.** Settled from reference data's `DATE_TO`,
  which is what it always was — `TestDelistingIsUnaffected` exists because the
  issue claimed disposal depended on the price read returning None, and it
  does not.

Prices are flat everywhere below so that any movement in a NAV or a holding is
the mechanism under test rather than the market.
"""
import pandas as pd
import pytest

from beacon.backtest.engine import BacktestEngine
from beacon.backtest.result import BacktestResult
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import CalculationError
from beacon.portfolio.base import Transaction
from beacon.server.backtests import assemble_result
from beacon.server.schemas import BacktestResultSummary
from beacon.testing import index_result_from_weights

START = "2021-01-04"
END = "2021-01-29"

# A business day the New York exchange was shut: Martin Luther King Jr. Day.
# No weekend rule reaches it, which is the point — a Monday-to-Friday range
# contains it and a calendar does not.
HOLIDAY = pd.Timestamp("2021-01-18")

# An ordinary Wednesday session, used as the day a bar is missing when it
# should not be.
OPEN_DAY = pd.Timestamp("2021-01-20")

PRICES = {"AAA": 100.0, "BBB": 50.0}


def build_fetcher(missing: dict[str, set[pd.Timestamp]] | None = None,
                  delist: dict[str, str] | None = None,
                  stop_prices: dict[str, str] | None = None) -> DataFetcher:
    """Two flat names over January 2021, with holes punched where asked.

    Args:
        missing: identifier -> dates that name prints no bar on.
        delist: identifier -> its reference-data `DATE_TO`.
        stop_prices: identifier -> the last date it prints a bar at all.

    Returns:
        DataFetcher: Market and reference data over the two names. Nothing
        prints on `HOLIDAY`, which is what a real store looks like: the
        exchange was shut, so no name has a bar.
    """
    missing = missing or {}
    delist = delist or {}
    stop_prices = stop_prices or {}

    rows = []

    for identifier, price in PRICES.items():
        for date in pd.bdate_range(START, END):
            stops = stop_prices.get(identifier)

            if date == HOLIDAY or date in missing.get(identifier, set()):
                continue

            if stops is not None and date > pd.Timestamp(stops):
                continue

            rows.append({"IDENTIFIER": identifier, "DATE": date,
                         "OPEN": price, "HIGH": price, "LOW": price,
                         "CLOSE": price, "VOLUME": 1_000.0})

    reference = pd.DataFrame([
        {"IDENTIFIER": identifier, "DATE_FROM": START,
         "DATE_TO": (pd.Timestamp(delist[identifier])
                     if identifier in delist else pd.NaT),
         "NAME": identifier, "CURRENCY": "USD", "EXCHANGE": "XNAS"}
        for identifier in PRICES
    ])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def build_engine(fetcher: DataFetcher,
                 rebalance: pd.Timestamp,
                 calendar: str | None = "XNYS",
                 end: str = END) -> BacktestEngine:
    """An engine holding both names equally from *rebalance* onward."""
    schedule = {rebalance: {"AAA": 0.5, "BBB": 0.5}}

    return BacktestEngine(start_date=START,
                          end_date=end,
                          initial_capital=100_000.0,
                          data_provider=fetcher,
                          index_result=index_result_from_weights(schedule),
                          calendar=calendar)


def transactions_on(result: BacktestResult,
                    date: pd.Timestamp) -> list[Transaction]:
    """Every transaction the run booked on *date*."""
    return [transaction for transaction in result.portfolio.transactions
            if pd.Timestamp(transaction.transaction_date) == date]


class TestAClosedDayRebalanceTrades:
    """The reported bug, in the shape it was reported."""

    def test_the_rebalance_executes(self):
        """Before BN-183 this was zero: no bar on the day, no price, no
        instruction, and a rebalance in the record that moved nothing."""
        result = build_engine(build_fetcher(), HOLIDAY).run()

        assert len(transactions_on(result, HOLIDAY)) == 2

    def test_the_portfolio_holds_what_the_schedule_asked_for(self):
        """The weights land where they were aimed: half the capital in each
        name, at the prices of the session that was in force."""
        result = build_engine(build_fetcher(), HOLIDAY).run()
        holdings = result.portfolio.holdings

        assert holdings["AAA"].quantity == pytest.approx(500.0)
        assert holdings["BBB"].quantity == pytest.approx(1_000.0)

    def test_the_run_says_which_session_it_priced_from(self):
        """Item 3 of the issue: the record states what it priced from rather
        than leaving a reader to infer it from the schedule."""
        result = build_engine(build_fetcher(), HOLIDAY).run()

        assert len(result.rebalance_pricing) == 1
        assert result.rebalance_pricing[0].date == HOLIDAY
        assert result.rebalance_pricing[0].priced_from == pd.Timestamp("2021-01-15")

    def test_an_ordinary_rebalance_prices_from_its_own_date(self):
        """Which is what makes the unequal pair above worth reading."""
        ordinary = pd.Timestamp("2021-01-06")
        result = build_engine(build_fetcher(), ordinary).run()

        assert [(row.date, row.priced_from)
                for row in result.rebalance_pricing] == [(ordinary, ordinary)]

    def test_a_closed_day_is_not_reported_as_a_gap(self):
        """A holiday is not missing data. Reporting it would bury the days
        that are, which is the only reason the report is worth having."""
        result = build_engine(build_fetcher(), HOLIDAY).run()

        assert result.price_gaps == []


class TestAHoleInTheDataIsCarriedAndReported:
    """A name with no bar on a session the calendar says was open."""

    def test_the_run_survives_it(self):
        """A stale quote beats killing the run: the other 499 names in a real
        universe did nothing wrong."""
        fetcher = build_fetcher(missing={"BBB": {OPEN_DAY}})
        result = build_engine(fetcher, pd.Timestamp("2021-01-06")).run()

        assert not result.trading_nav.empty
        assert "BBB" in result.portfolio.holdings

    def test_the_mark_is_the_last_known_price(self):
        """Carried forward, so the NAV does not step on a missing bar."""
        fetcher = build_fetcher(missing={"BBB": {OPEN_DAY}})
        result = build_engine(fetcher, pd.Timestamp("2021-01-06")).run()
        nav = result.trading_nav

        assert nav.loc[OPEN_DAY] == pytest.approx(nav.loc[pd.Timestamp("2021-01-19")])

    def test_it_is_published_rather_than_absorbed(self):
        fetcher = build_fetcher(missing={"BBB": {OPEN_DAY}})
        result = build_engine(fetcher, pd.Timestamp("2021-01-06")).run()

        assert [(gap.asset_id, gap.date, gap.priced_from)
                for gap in result.price_gaps] == [
            ("BBB", OPEN_DAY, pd.Timestamp("2021-01-19"))]

    def test_a_second_run_reports_its_own_gaps_only(self):
        """The engine is reusable, and a report that accumulated across runs
        would double-count the same missing bar."""
        fetcher = build_fetcher(missing={"BBB": {OPEN_DAY}})
        engine = build_engine(fetcher, pd.Timestamp("2021-01-06"))
        engine.run()

        assert len(engine.run().price_gaps) == 1

    def test_one_missing_bar_is_one_gap(self):
        """A rebalance day prices each name several times — the mark, the sell
        test, the buy test, the re-mark — and one absence is one fact."""
        fetcher = build_fetcher(missing={"BBB": {OPEN_DAY}})
        result = build_engine(fetcher, OPEN_DAY).run()

        assert len(result.price_gaps) == 1

    def test_the_holiday_beside_it_still_is_not_one(self):
        """Both absences in one run, told apart by the calendar alone."""
        fetcher = build_fetcher(missing={"BBB": {OPEN_DAY}})
        result = build_engine(fetcher, pd.Timestamp("2021-01-06")).run()

        assert {gap.date for gap in result.price_gaps} == {OPEN_DAY}

    def test_a_past_date_is_not_answered_from_a_later_close(self):
        """The run walks forward and the last bar per name is cached to keep a
        long absence cheap. Asked about a date behind that cache — which only
        happens off the run's own path — the scan restarts rather than handing
        back a price from the future."""
        rows = []

        for step, date in enumerate(pd.bdate_range(START, END)):
            if date in (HOLIDAY, OPEN_DAY):
                continue

            rows.append({"IDENTIFIER": "BBB", "DATE": date, "OPEN": 50.0 + step,
                         "HIGH": 50.0 + step, "LOW": 50.0 + step,
                         "CLOSE": 50.0 + step, "VOLUME": 1_000.0})

        reference = pd.DataFrame([{"IDENTIFIER": "BBB", "DATE_FROM": START,
                                   "DATE_TO": pd.NaT, "NAME": "B",
                                   "CURRENCY": "USD", "EXCHANGE": "XNAS"}])
        fetcher = DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                              ReferenceData.from_dataframe(reference))

        engine = BacktestEngine(start_date=START, end_date=END,
                                initial_capital=100_000.0,
                                data_provider=fetcher,
                                index_result=index_result_from_weights(
                                    {pd.Timestamp("2021-01-06"): {"BBB": 1.0}}),
                                calendar="XNYS")
        engine.run()

        # 2021-01-19 is the twelfth business day of the window, so its close is
        # 50 + 11 -- while the run's last bar, 2021-01-29, is 50 + 19.
        assert engine._fetch_price("BBB", OPEN_DAY) == pytest.approx(61.0)

    def test_without_a_calendar_the_data_s_own_sessions_stand_in(self):
        """A hand-assembled engine has no calendar to ask. A day other names
        printed on is still a day this one's silence is its own, which keeps
        the distinction available — less precisely — to a library caller."""
        fetcher = build_fetcher(missing={"BBB": {OPEN_DAY}})
        result = build_engine(fetcher, pd.Timestamp("2021-01-06"),
                              calendar=None).run()

        assert [gap.date for gap in result.price_gaps] == [OPEN_DAY]


class TestPastTheDataItRefuses:
    """BN-179's bound, in BN-179's words: backfill within, refuse beyond."""

    def test_the_run_raises(self):
        """Not a stale print wearing a current date. Nothing is known out
        there, and a carried price would answer a different question."""
        engine = build_engine(build_fetcher(), pd.Timestamp("2021-01-06"),
                              end="2021-02-05")

        with pytest.raises(CalculationError):
            engine.run()

    def test_the_message_names_the_date_and_the_data_s_end(self):
        """"Ask for an earlier date or refresh the store" is the action, and
        neither half of it is discoverable from the failure otherwise."""
        engine = build_engine(build_fetcher(), pd.Timestamp("2021-01-06"),
                              end="2021-02-05")

        with pytest.raises(CalculationError) as raised:
            engine.run()

        assert "2021-02-01" in str(raised.value)
        assert "2021-01-29" in str(raised.value)


class TestDelistingIsUnaffected:
    """The issue's central claim, corrected: disposal never read the price.

    It claimed `_dispose_delisted` depended on `_fetch_price` returning None
    once a name's rows stopped, and that the price read therefore could not
    change until a separate delisting signal existed. The signal already
    existed — `DataFetcher.delisting_dates()`, resolved from reference data's
    `DATE_TO` — so these pin it rather than trusting the reading.
    """

    def test_the_signal_is_reference_data(self):
        """Prices run to the end of the window and the name is still retired,
        which no mechanism reading absent bars could manage."""
        fetcher = build_fetcher(delist={"BBB": "2021-01-20"})

        assert fetcher.delisting_dates() == {"BBB": pd.Timestamp("2021-01-20")}

    def test_a_delisted_name_is_settled_into_cash(self):
        fetcher = build_fetcher(delist={"BBB": "2021-01-20"})
        result = build_engine(fetcher, pd.Timestamp("2021-01-06")).run()

        assert "BBB" not in result.portfolio.holdings
        assert [transaction.transaction_type
                for transaction in transactions_on(result,
                                                   pd.Timestamp("2021-01-21"))
                if transaction.asset_id == "BBB"] == ["SELL"]

    def test_it_is_settled_when_its_prices_stop_too(self):
        """The ordinary case, where the rows end with the listing."""
        fetcher = build_fetcher(delist={"BBB": "2021-01-20"},
                                stop_prices={"BBB": "2021-01-20"})
        result = build_engine(fetcher, pd.Timestamp("2021-01-06")).run()

        assert "BBB" not in result.portfolio.holdings

    def test_no_price_is_carried_past_a_delisting(self):
        """The risk the owner named that the issue did not: an unbounded
        backfill would trade a dead name forever at its last close. The
        delisting bounds the carry, so there is no price rather than a stale
        one — and no gap either, since nothing is missing."""
        fetcher = build_fetcher(delist={"BBB": "2021-01-20"},
                                stop_prices={"BBB": "2021-01-20"})
        engine = build_engine(fetcher, pd.Timestamp("2021-01-06"))
        result = engine.run()

        assert engine._fetch_price("BBB", pd.Timestamp("2021-01-25")) is None
        assert [gap.asset_id for gap in result.price_gaps] == []


class TestTheGapsReachTheWire:
    """Published, or the report is a fact the library keeps to itself."""

    def test_the_record_carries_them(self):
        fetcher = build_fetcher(missing={"BBB": {OPEN_DAY}})
        result = build_engine(fetcher, pd.Timestamp("2021-01-06")).run()
        record = BacktestResultSummary.from_result(result)

        assert [(row.asset_id, row.date, row.priced_from)
                for row in record.price_gaps] == [
            ("BBB", "2021-01-20", "2021-01-19")]

    def test_the_run_payload_carries_them(self):
        fetcher = build_fetcher(missing={"BBB": {OPEN_DAY}})
        result = build_engine(fetcher, pd.Timestamp("2021-01-06")).run()
        book = result.index.tracked
        assert book is not None and book.source is not None

        payload = assemble_result(result, book.source)

        assert [row.date for row in payload.price_gaps] == ["2021-01-20"]

    def test_both_publish_what_each_rebalance_priced_from(self):
        fetcher = build_fetcher()
        result = build_engine(fetcher, HOLIDAY).run()
        book = result.index.tracked
        assert book is not None and book.source is not None

        record = BacktestResultSummary.from_result(result)
        payload = assemble_result(result, book.source)
        expected = [("2021-01-18", "2021-01-15")]

        assert [(row.date, row.priced_from)
                for row in record.rebalance_pricing] == expected
        assert [(row.date, row.priced_from)
                for row in payload.rebalance_pricing] == expected


class TestAFailedDelistingLookupIsNotAnEmptyOne:
    """BN-197. The engine used to swallow the failure into `{}` and carry on.

    Empty is not "unknown" here — it is "nothing is ever delisted", and every
    reader acts on it. `_fetch_price` stops declining to carry a dead name
    forward and `_dispose_delisted` never settles, so the run finishes holding
    names that no longer exist, each marked at its last close, and publishes a
    NAV as though that were real. The only record was a WARNING.

    `IndexCalculator.delisting_schedule` calls the same method bare and always
    has, so the two surfaces gave different answers to one failure.
    """

    def test_the_failure_propagates_rather_than_reading_as_no_delistings(self):
        fetcher = build_fetcher(delist={"BBB": "2021-01-20"})
        fetcher.delisting_dates = _raising(ValueError("malformed DATE_TO"))
        engine = build_engine(fetcher, pd.Timestamp("2021-01-06"))

        with pytest.raises(ValueError, match="malformed DATE_TO"):
            engine.run()

    def test_it_does_not_finish_holding_a_name_that_stopped_existing(self):
        """The harm, stated as the outcome rather than the mechanism.

        Under the old behaviour this run completed with BBB still held and
        marked at its last close for every day after it was delisted.
        """
        fetcher = build_fetcher(delist={"BBB": "2021-01-20"},
                                stop_prices={"BBB": "2021-01-20"})
        fetcher.delisting_dates = _raising(KeyError("DATE_TO"))
        engine = build_engine(fetcher, pd.Timestamp("2021-01-06"))

        with pytest.raises(KeyError):
            engine.run()

    def test_a_provider_without_the_method_is_still_fine(self):
        """The case the `getattr` exists for, and the one that must not move.

        A hand-assembled provider need not implement this, and a backtest over
        a universe where nothing is delisted should not require it to. That is
        an absent capability, not a failed lookup.
        """
        fetcher = _WithoutDelistings(build_fetcher())
        engine = build_engine(fetcher, pd.Timestamp("2021-01-06"))

        assert engine._delisting_dates() == {}
        assert engine.run().trading_nav.notna().all()

    def test_a_non_mapping_answer_still_reads_as_nothing_leaves(self):
        """Also an answer rather than a failure: a double returning a stand-in."""
        fetcher = build_fetcher()
        fetcher.delisting_dates = lambda: None
        engine = build_engine(fetcher, pd.Timestamp("2021-01-06"))

        assert engine._delisting_dates() == {}

    def test_the_engine_and_the_calculator_agree_about_one_failure(self):
        """The point of the issue: two surfaces, one call, one answer.

        Neither number is asserted here — only that the two stopped
        disagreeing, which is the thing that was wrong.
        """
        from beacon.index.calculation import IndexCalculator

        fetcher = build_fetcher()
        fetcher.delisting_dates = _raising(ValueError("malformed DATE_TO"))

        engine_raised = _raises(
            build_engine(fetcher, pd.Timestamp("2021-01-06"))._delisting_dates)
        calculator_raised = _raises(
            lambda: IndexCalculator.delisting_schedule(
                _WithData(fetcher)))

        assert engine_raised is calculator_raised is True


def _raising(error: BaseException):
    """A `delisting_dates` stand-in that fails the way a bad store would."""
    def _fail() -> dict[str, pd.Timestamp]:
        raise error

    return _fail


def _raises(call) -> bool:
    """Whether *call* raised, without caring which exception it was."""
    try:
        call()
    except Exception:
        return True

    return False


class _WithData:
    """The one attribute `delisting_schedule` reads off its calculator."""

    def __init__(self,
                 data: DataFetcher):
        self.data = data


class _WithoutDelistings:
    """A provider that does not offer `delisting_dates` at all.

    `del fetcher.delisting_dates` cannot express this — the name is on the
    class, not the instance — and a provider genuinely lacking the method is
    the case the `getattr` guard exists for, so it has to be a real object
    rather than a patched one.
    """

    def __init__(self,
                 fetcher: DataFetcher):
        self._fetcher = fetcher

    def __getattr__(self,
                    name: str):
        if name == "delisting_dates":
            raise AttributeError(name)

        return getattr(self._fetcher, name)
