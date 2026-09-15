# tests/test_session_agreement.py
"""BN-186: the data and the daily loop agree about which days exist.

This is the property nothing could test before. BN-180 made the *schedule*
calendar-aware and left two things counting Monday to Friday — the daily loop
that publishes a level, and the generator that produces the bars it reads. So
they agreed, and agreed wrongly: every generated fixture carried a bar on
25 December, and an index calculated over it published a level there. A defect
whose trigger was a closed market could not be expressed, let alone caught.

The tests here are deliberately about *agreement* rather than about either
side. Each one names a day a real exchange was shut and asserts that neither
the panel nor the index has anything to say about it — and, so that the
calendar is demonstrably an input rather than decoration, that generating on
another venue moves the gaps to that venue's holidays.
"""
import pandas as pd
import pytest

from beacon.backtest.engine import BacktestEngine
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.index.schedule import sessions
from beacon.synthetic.dataset import SyntheticConfig, generate
from beacon.testing import dataset

START = "2024-01-01"
END = "2024-12-31"
NEW_YORK = "XNYS"
LONDON = "XLON"

# Weekdays in 2024 that separate the two venues. Independence Day is a New
# York closure and an ordinary London session; Boxing Day is the reverse.
# Neither is reachable by a weekend rule, which is the point of choosing them.
ONLY_LONDON_TRADES = pd.Timestamp("2024-07-04")
ONLY_NEW_YORK_TRADES = pd.Timestamp("2024-12-26")

# Every US market holiday of 2024 that falls on a weekday.
NEW_YORK_CLOSURES = (
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
)


def build(calendar: str) -> SyntheticConfig:
    """A small generated universe on one venue's calendar.

    No listings and no delistings, so every name spans the whole panel and a
    missing bar can only be a day the panel does not have. Turnover is a real
    part of the generator and a distraction here: it would make a name's dates
    a subset of the sessions for a reason that has nothing to do with
    calendars.
    """
    return SyntheticConfig(assets=6, start=START, end=END, seed=7,
                           features=False, calendar=calendar,
                           delisting_rate=0.0, listing_rate=0.0)


@pytest.fixture(scope="module")
def new_york():
    return generate(build(NEW_YORK))


@pytest.fixture(scope="module")
def london():
    return generate(build(LONDON))


def dates_of(generated,
             identifier: str) -> pd.DatetimeIndex:
    """The dates *identifier* printed a bar on."""
    rows = generated.market.data.xs(identifier, level="IDENTIFIER")

    return pd.DatetimeIndex(rows.index)


class TestTheGeneratorProducesSessions:
    """Half the agreement: the data side."""

    def test_the_panel_is_exactly_the_calendars_sessions(self, new_york):
        expected = sessions(pd.Timestamp(START), pd.Timestamp(END), NEW_YORK)
        name = sorted(generated_names(new_york))[0]

        assert list(dates_of(new_york, name)) == list(expected)

    @pytest.mark.parametrize("closed", NEW_YORK_CLOSURES)
    def test_no_name_has_a_bar_on_a_day_the_market_was_shut(self,
                                                            new_york,
                                                            closed):
        """All ten fall on a weekday, so `bdate_range` put a bar on every one
        of them and no weekend rule would ever have removed it."""
        day = pd.Timestamp(closed)
        frame = new_york.market.data

        assert day not in frame.index.get_level_values("DATE")

    def test_the_calendar_is_an_input_not_a_decoration(self,
                                                       new_york,
                                                       london):
        """Generated on London, the gaps move to London's holidays."""
        us_name = sorted(generated_names(new_york))[0]
        uk_name = sorted(generated_names(london))[0]

        us_dates = set(dates_of(new_york, us_name))
        uk_dates = set(dates_of(london, uk_name))

        assert ONLY_LONDON_TRADES not in us_dates
        assert ONLY_LONDON_TRADES in uk_dates
        assert ONLY_NEW_YORK_TRADES in us_dates
        assert ONLY_NEW_YORK_TRADES not in uk_dates

    def test_an_unusable_calendar_is_refused_rather_than_ignored(self):
        with pytest.raises(ValueError, match="not a trading calendar"):
            SyntheticConfig(assets=2, start=START, end=END, calendar="NOPE")


class TestTheLoopPublishesSessions:
    """The other half: an index has a level on a session and on nothing else."""

    @pytest.fixture(scope="class")
    def result(self):
        return IndexCalculator(definition(NEW_YORK),
                               dataset.data_fetcher()).run(
            start_date=dataset.START, end_date=dataset.END)

    def test_the_levels_are_exactly_the_sessions(self, result):
        expected = sessions(pd.Timestamp(dataset.START),
                            pd.Timestamp(dataset.END), NEW_YORK)

        assert list(result.index_levels.index) == list(expected)

    @pytest.mark.parametrize("closed", NEW_YORK_CLOSURES)
    def test_there_is_no_level_on_a_day_the_market_was_shut(self,
                                                            result,
                                                            closed):
        assert pd.Timestamp(closed) not in result.index_levels.index

    def test_the_daily_weights_panel_agrees_with_the_levels(self, result):
        """The panel is recorded as the loop walks, so a level and a weights
        row exist on exactly the same days or one of them is inventing dates.
        """
        recorded = pd.DatetimeIndex(sorted(set(result.daily_weights["DATE"])))

        assert list(recorded) == list(result.index_levels.index)


class TestTheBacktestAgreesWithBoth:
    """The third reader of the same calendar."""

    @pytest.fixture(scope="class")
    def run(self):
        index_result = IndexCalculator(definition(NEW_YORK),
                                       dataset.data_fetcher()).run(
            start_date=dataset.START, end_date=dataset.END)

        return BacktestEngine(start_date=dataset.START,
                              end_date=dataset.END,
                              initial_capital=1_000_000.0,
                              data_provider=dataset.data_fetcher(),
                              index_result=index_result,
                              calendar=NEW_YORK).run()

    def test_the_nav_is_dated_to_the_same_sessions(self, run):
        expected = sessions(pd.Timestamp(dataset.START),
                            pd.Timestamp(dataset.END), NEW_YORK)

        assert list(run.trading_nav.index) == list(expected)

    def test_nothing_is_reported_as_a_price_gap(self, run):
        """A holiday is not missing data. Every absent bar in the fixture is
        now a day the calendar says was closed, so the run has nothing to
        report — which is only true because the two were built from one
        calendar.
        """
        assert run.price_gaps == []


def generated_names(generated) -> set[str]:
    """The equity identifiers in a generated panel, excluding FX pairs."""
    return set(generated.universe.index)


def definition(calendar: str) -> IndexDefinition:
    """An equal-weighted index over the canonical universe."""
    return IndexDefinition(index_id="AGREE",
                           index_name="Agreement",
                           base_date=dataset.START,
                           base_value=1000.0,
                           currency="USD",
                           eligibility_rules=[],
                           weighting_scheme=EqualWeighted(),
                           rebalancing_frequency="QUARTERLY",
                           calendar=calendar,
                           universe_identifiers=list(dataset.UNIVERSE))
