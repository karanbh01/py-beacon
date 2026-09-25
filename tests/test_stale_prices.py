# tests/test_stale_prices.py
"""BN-211: dropping names that have stopped trading, if you want to.

Karan's ask, alongside BN-210's staleness reporting: the ability to drop stale
companies past a threshold in both index construction and backtests, with the
user choosing whether to keep securities on stale pricing or not.

A modelling choice rather than a data property, and **global** on his call --
one answer for the installation, reaching index construction and backtests
alike, the way `fx_policy` does. `None` keeps everything, which is what the
library always did, so adopting the setting moves nothing until asked.

The two halves share one definition of stale (`DataFetcher.stale_identifiers`)
rather than each deciding for itself. That is the rule BN-206 and BN-207 were
about: five conversion sites agreed on the day they were written and disagreed
a year later.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.backtest.engine import BacktestEngine
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.index.calculation import IndexCalculator
from beacon.index.calculation.selection import (
    STALENESS_POSITION,
    STALENESS_RULE_NAME,
)
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted, MarketCapRule
from beacon.server import ServerConfig, create_app
from beacon.testing import index_result_from_weights

TOKEN = "test-token-value"
START = "2025-01-02"
END = "2025-06-30"
AS_OF = pd.Timestamp(END)

# QUIET's last bar is 76 days before AS_OF: comfortably past a 30-day
# threshold, comfortably inside a 120-day one, so one fixture exercises both
# sides of the comparison.
QUIET_LAST_TRADED = "2025-04-15"


def build_fetcher(threshold: int | None = None,
                  include_ghost: bool = False) -> DataFetcher:
    """Two names trading and, optionally, one that never traded at all."""
    rows: list[dict[str, object]] = [
        {"IDENTIFIER": "LOUD", "DATE": date, "CLOSE": 100.0,
         "SHARES_OUTSTANDING": 1_000_000}
        for date in pd.bdate_range(START, END)]

    rows += [{"IDENTIFIER": "QUIET", "DATE": date, "CLOSE": 50.0,
              "SHARES_OUTSTANDING": 1_000_000}
             for date in pd.bdate_range(START, QUIET_LAST_TRADED)]

    names = ["LOUD", "QUIET"] + (["GHOST"] if include_ghost else [])
    reference = pd.DataFrame([{"IDENTIFIER": name, "DATE_FROM": "2020-01-01",
                               "NAME": name, "CURRENCY": "USD",
                               "EXCHANGE": "XNYS"}
                              for name in names])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference),
                       max_price_staleness_days=threshold)


def definition(rules: list | None = None) -> IndexDefinition:
    return IndexDefinition(index_id="S",
                           index_name="Staleness",
                           base_date=START,
                           base_value=1000.0,
                           currency="USD",
                           eligibility_rules=rules or [],
                           weighting_scheme=EqualWeighted(),
                           rebalancing_frequency="MONTHLY",
                           calendar="XNYS",
                           universe_identifiers=["LOUD", "QUIET"])


def selection(fetcher: DataFetcher):
    """The selection funnel at AS_OF, provenance and all."""
    calculator = IndexCalculator(definition(), fetcher)

    return calculator.select_with_provenance(
        calculator.resolve_universe(AS_OF), AS_OF)


def engine_over(fetcher: DataFetcher) -> BacktestEngine:
    schedule = {AS_OF: {"LOUD": 0.5, "QUIET": 0.5}}

    return BacktestEngine(start_date=START,
                          end_date=END,
                          initial_capital=100_000.0,
                          data_provider=fetcher,
                          index_result=index_result_from_weights(schedule),
                          calendar="XNYS")


class TestTheSettingIsOptOut:
    """Nothing moves until somebody asks it to."""

    def test_the_default_is_no_threshold(self):
        assert build_fetcher().max_price_staleness_days is None

    def test_a_stale_name_is_kept_by_default(self):
        survivors = selection(build_fetcher()).survivors

        assert [asset.asset_id for asset in survivors] == ["LOUD", "QUIET"]

    def test_the_funnel_gains_no_rung_by_default(self):
        """An index with no threshold must produce exactly the provenance it
        always did, or every client reading the funnel by position moves."""
        steps = selection(build_fetcher()).steps

        assert [step.position for step in steps] == [0]

    def test_nothing_is_read_when_no_threshold_is_set(self):
        """The gate costs one comparison, not a batch market read."""
        assert build_fetcher().stale_identifiers(["LOUD", "QUIET"],
                                                 AS_OF) == set()

    def test_a_threshold_below_a_day_is_refused(self):
        """Zero would drop every name that did not trade today, which is not
        a threshold anybody means. Treating it silently as "off" would be
        worse: the caller asked for something."""
        with pytest.raises(ValueError, match="at least 1 day"):
            build_fetcher(threshold=0)


class TestIndexConstruction:

    def test_a_stale_name_is_dropped(self):
        survivors = selection(build_fetcher(threshold=30)).survivors

        assert [asset.asset_id for asset in survivors] == ["LOUD"]

    def test_a_generous_threshold_keeps_it(self):
        """The comparison is against the threshold, not against "recently"."""
        survivors = selection(build_fetcher(threshold=120)).survivors

        assert [asset.asset_id for asset in survivors] == ["LOUD", "QUIET"]

    def test_the_funnel_records_why(self):
        """A name that vanishes without a reason is what the provenance
        record exists to prevent."""
        steps = selection(build_fetcher(threshold=30)).steps
        rung = next(step for step in steps
                    if step.position == STALENESS_POSITION)

        assert rung.rule_name == STALENESS_RULE_NAME
        assert rung.excluded == ["QUIET"]
        assert rung.remaining == 1

    def test_each_exclusion_names_its_own_rung(self):
        """With the stale rung in the funnel, a name a rule removed is still
        attributed to that rule, and a stale name to the stale rung."""
        rule = MarketCapRule(min_market_cap=1e9)
        calculator = IndexCalculator(definition(rules=[rule]),
                                     build_fetcher(threshold=30))
        result = calculator.select_with_provenance(
            calculator.resolve_universe(AS_OF), AS_OF)

        assert result.excluded_by("QUIET").position == STALENESS_POSITION
        assert result.excluded_by("LOUD").rule_name == "MarketCapRule"

    def test_the_rung_sits_outside_the_rule_positions(self):
        """Negative so it cannot be confused with a rule's position, and so a
        consumer mapping positions onto the definition's rules has to handle
        it deliberately rather than read the wrong rule's id."""
        assert STALENESS_POSITION < 0

    def test_a_name_that_never_traded_is_not_called_stale(self):
        """A different condition with a different remedy. Folding the two
        together would excuse a missing instrument as a quiet one."""
        fetcher = build_fetcher(threshold=30, include_ghost=True)

        assert fetcher.stale_identifiers(["LOUD", "QUIET", "GHOST"],
                                         AS_OF) == {"QUIET"}


class TestBacktests:

    def test_a_stale_name_is_dropped_from_the_target(self):
        target = engine_over(build_fetcher(threshold=30))._drop_stale(
            {"LOUD": 0.5, "QUIET": 0.5}, AS_OF)

        assert set(target) == {"LOUD"}

    def test_the_remaining_weights_are_renormalised(self):
        """Leaving the survivors at 0.5 would put the difference into cash
        silently and report a tracking gap against an index that holds the
        name -- a smaller book reported as a different one."""
        target = engine_over(build_fetcher(threshold=30))._drop_stale(
            {"LOUD": 0.5, "QUIET": 0.5}, AS_OF)

        assert target["LOUD"] == pytest.approx(1.0)

    def test_an_untouched_target_is_returned_unchanged(self):
        weights = {"LOUD": 0.5, "QUIET": 0.5}

        assert engine_over(build_fetcher())._drop_stale(weights,
                                                        AS_OF) == weights

    def test_a_run_completes_holding_only_the_live_name(self):
        """End to end, because the target filter and the book are different
        things and only the second is what a user sees."""
        result = engine_over(build_fetcher(threshold=30)).run()

        assert "QUIET" not in result.portfolio.holdings


class TestTheSettingIsPublished:
    """It changes index membership, so a reader has to be able to see it."""

    def client_with(self,
                    threshold: int | None,
                    tmp_path) -> TestClient:
        config = ServerConfig(auth_token=TOKEN,
                              data_fetcher=build_fetcher(threshold),
                              storage_root=tmp_path)

        return TestClient(create_app(config))

    def health(self,
               client: TestClient) -> dict:
        return client.get("/health",
                          headers={"Authorization": f"Bearer {TOKEN}"}).json()

    def test_it_appears_on_health(self,
                                  tmp_path):
        body = self.health(self.client_with(30, tmp_path))

        assert body["max_price_staleness_days"] == 30

    def test_no_threshold_reports_null(self,
                                       tmp_path):
        body = self.health(self.client_with(None, tmp_path))

        assert body["max_price_staleness_days"] is None
