# tests/test_read_volume.py
"""BN-212: how many times a run asks the data layer for a price.

Not a timing test. Timings are machine-dependent and flap, and a threshold
loose enough never to flap is loose enough never to catch anything. What went
wrong here is *countable*: the index calculator and the backtest engine each
read one name on one date at a time, so a 200-name three-year run made about
172,000 separate frame slices where 782 would do. Each slice costs what the
whole frame costs rather than what one row does, which is why the cost grew
with the universe rather than with the work.

The session panel that fixes it has existed since BN-190. It was wired into
selection and weighting -- which run at rebalances, roughly forty times over
this loop's eight hundred -- and never into the daily valuation, which is the
one that runs every session. `fetch_price` consults the panel; the
`fetch_market_data` both hot paths called does not.

So these tests count reads and assert the *shape*: per session, not per name
per session. A regression here is a number that grows with the universe.
"""
import pandas as pd
import pytest

from beacon.backtest.engine import BacktestEngine
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted

START = "2024-01-02"
END = "2024-03-28"
SESSIONS = len(pd.bdate_range(START, END))


class CountingMarketData(MarketData):
    """Market data that records how often the frame is actually sliced.

    Counting at `MarketData.get` rather than at `DataFetcher.fetch_market_data`
    is the whole point, and getting it wrong was instructive: the first version
    of this file counted the fetcher method and read **zero**, because
    `fetch_price` and `warm_session` both reach the frame directly. Counting
    the public wrapper would have measured a method the hot paths no longer
    call, and passed no matter how badly they behaved.

    `get` is the expensive operation -- one slice of a MultiIndexed frame,
    costing what the frame costs rather than what a row does -- so it is the
    thing worth bounding.
    """

    # A class attribute rather than one set in `__init__`: `from_dataframe` is
    # a classmethod that does not route through it, so an instance built the
    # ordinary way had no counter at all. `self.slices += 1` reads this and
    # writes an instance attribute, so the tally is still per-instance.
    slices = 0

    def get(self,
            identifier,
            start_date=None,
            end_date=None,
            columns=None):
        self.slices += 1

        return super().get(identifier, start_date, end_date, columns)


class CountingFetcher(DataFetcher):
    """A real fetcher over counting market data.

    Subclassed rather than mocked: the panel is the thing under test, and a
    double would have to reimplement it to be worth counting -- at which point
    the count would be a fact about the double.
    """

    @property
    def market_reads(self) -> int:
        """Frame slices taken so far."""
        return self._market.slices  # type: ignore[attr-defined]


def build(names: list[str]) -> CountingFetcher:
    """Flat prices, so nothing below is about the market moving."""
    rows = [{"IDENTIFIER": name, "DATE": date, "CLOSE": 100.0,
             "VOLUME": 1_000_000.0, "SHARES_OUTSTANDING": 1_000_000}
            for name in names
            for date in pd.bdate_range(START, END)]
    reference = pd.DataFrame([{"IDENTIFIER": name, "DATE_FROM": "2020-01-01",
                               "NAME": name, "CURRENCY": "USD",
                               "EXCHANGE": "XNYS"}
                              for name in names])

    market = CountingMarketData.from_dataframe(pd.DataFrame(rows))

    return CountingFetcher(market, ReferenceData.from_dataframe(reference))


def definition(names: list[str]) -> IndexDefinition:
    return IndexDefinition(index_id="READS",
                           index_name="Read Volume",
                           base_date=START,
                           base_value=1000.0,
                           currency="USD",
                           eligibility_rules=[],
                           weighting_scheme=EqualWeighted(),
                           rebalancing_frequency="MONTHLY",
                           calendar="XNYS",
                           universe_identifiers=names)


def index_reads(count: int) -> int:
    """Frame slices taken by an index calculation over *count* names."""
    names = [f"N{index:03d}" for index in range(count)]
    fetcher = build(names)
    IndexCalculator(definition(names), fetcher).run(end_date=END)

    return fetcher.market_reads


def backtest_reads(count: int) -> int:
    """Frame slices taken by a backtest over *count* names."""
    names = [f"N{index:03d}" for index in range(count)]
    fetcher = build(names)
    result = IndexCalculator(definition(names), fetcher).run(end_date=END)

    before = fetcher.market_reads
    BacktestEngine(start_date=START,
                   end_date=END,
                   initial_capital=1_000_000.0,
                   data_provider=fetcher,
                   index_result=result,
                   calendar="XNYS").run()

    return fetcher.market_reads - before


# What a per-name-per-session read costs, which is the shape being forbidden.
# The bound is a quarter of it: comfortably above what either path does now,
# and far below what either did before, so the test fails on the regression
# rather than on ordinary drift.
def per_name_per_session(names: int) -> int:
    return names * SESSIONS


class TestTheIndexCalculationDoesNotReadPerName:
    """Measured: 40 names over 63 sessions takes 300 slices, against the 2,520
    a per-name-per-day read needs as a floor."""

    def test_reads_are_not_per_name_per_session(self):
        reads = index_reads(40)
        ceiling = per_name_per_session(40) // 4

        assert reads < ceiling, (
            f"{reads} slices for 40 names over {SESSIONS} sessions is "
            f"approaching the {per_name_per_session(40)} a per-name-per-day "
            f"read costs; the daily valuation is reading name by name again")

    def test_growth_is_by_rebalance_rather_than_by_session(self):
        """Four times the universe must not mean four times the reads.

        It does not mean *one* times either, and pinning it at one would be
        wrong: what remains scales with rebalances, which a monthly index has
        three of over this window against sixty-three sessions. Selection and
        weighting read per name on the days they run, and those reads are the
        residual. The daily valuation is what no longer does.
        """
        small = index_reads(10)
        large = index_reads(40)

        assert large < small * 3, (
            f"reads grew {large / small:.1f}x for 4x the universe; above 3x "
            f"the daily loop is scaling with the universe again")


class TestTheBacktestDoesNotReadPerName:
    """Measured: 40 names over 63 sessions takes 101 slices."""

    def test_reads_are_not_per_name_per_session(self):
        reads = backtest_reads(40)
        ceiling = per_name_per_session(40) // 4

        assert reads < ceiling, (
            f"{reads} slices for 40 names over {SESSIONS} sessions is "
            f"approaching the {per_name_per_session(40)} a per-name-per-day "
            f"read costs; the engine is pricing name by name again")

    def test_growth_is_by_rebalance_rather_than_by_session(self):
        small = backtest_reads(10)
        large = backtest_reads(40)

        assert large < small * 3, (
            f"reads grew {large / small:.1f}x for 4x the book")


class TestTheNumbersDoNotMove:
    """The point of the panel: it changes when an answer arrives, never what
    the answer is. A faster run that computes something else is not faster."""

    def test_an_index_level_is_unchanged_by_warming(self):
        names = [f"N{index:03d}" for index in range(10)]

        warm = IndexCalculator(definition(names), build(names)).run(end_date=END)

        cold_fetcher = build(names)
        cold_fetcher.warm_session = None  # type: ignore[method-assign]
        cold = IndexCalculator(definition(names), cold_fetcher).run(end_date=END)

        assert warm.index_levels.iloc[-1] == pytest.approx(
            cold.index_levels.iloc[-1])

    def test_a_provider_without_the_hint_still_runs(self):
        """`warm_session` is an optimisation, not a contract. A hand-assembled
        provider that lacks it must still produce an index."""
        names = [f"N{index:03d}" for index in range(5)]
        fetcher = build(names)
        fetcher.warm_session = None  # type: ignore[method-assign]

        result = IndexCalculator(definition(names), fetcher).run(end_date=END)

        assert not result.index_levels.empty
