# tests/test_index_eligibility.py
"""BN-182: an eligibility rule resolves its date, or refuses — it never excludes.

BN-179 taught the market-cap *weighting* to resolve a requested date back to
the session actually in force. The same exact-date read survived one layer up
in `MarketCapRule`, where its consequence was worse: on a closed day every name
failed, the universe emptied, and the fixed weighting was handed nothing to
weight. An index of no constituents still computes.

The dates here are the issue's own reproduction, kept verbatim so the file
reads as the bug report it closes.
"""
import ast
import inspect
import textwrap

import pandas as pd
import pytest

from beacon.asset.base import Asset
from beacon.asset.equity import Equity
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import CalculationError
from beacon.index.calculation.selection import select_with_provenance
from beacon.index.methodology import LiquidityRule, MarketCapRule

START = "2026-08-03"
END = "2026-08-31"
DATES = pd.bdate_range(START, END)

LAST_BAR = pd.Timestamp("2026-08-31")
SATURDAY_INSIDE_THE_DATA = pd.Timestamp("2026-08-29")
THE_FRIDAY_BEFORE = pd.Timestamp("2026-08-28")
PAST_THE_END = pd.Timestamp("2026-09-11")

# AAA clears a 50k floor comfortably; ZZZ never does. Shares are equal, so the
# price is the whole story and an exclusion can only be about the cap.
PRICES = {"AAA": 100.0, "ZZZ": 1.0}
SHARES = 1_000
VOLUME = 1_000_000


def build_fetcher(prices: dict[str, float] | None = None) -> DataFetcher:
    """Flat prices on business days only, so weekends are genuinely absent."""
    closes = prices or PRICES
    market = pd.DataFrame([
        {"IDENTIFIER": name,
         "DATE": date,
         "CLOSE": price,
         "VOLUME": VOLUME,
         "SHARES_OUTSTANDING": SHARES}
        for name, price in closes.items()
        for date in DATES
    ])
    reference = pd.DataFrame([
        {"IDENTIFIER": name, "DATE_FROM": "2020-01-01", "NAME": name,
         "CURRENCY": "USD", "EXCHANGE": "NYSE"}
        for name in closes
    ])

    return DataFetcher(MarketData.from_dataframe(market),
                       ReferenceData.from_dataframe(reference))


def equity(ticker: str) -> Equity:
    """One named equity."""
    return Equity(name=f"{ticker} Corp", currency="USD",
                  ticker=ticker, exchange="NYSE")


def universe() -> list[Asset]:
    """Both names, in a fixed order."""
    return [equity(ticker) for ticker in PRICES]


class Boom(MarketCapRule):
    """A market-cap rule whose data source is broken rather than empty."""

    def is_eligible(self,
                    asset,
                    current_date,
                    market_data_provider,
                    context=None) -> bool:
        raise RuntimeError("the data source is down")


class TestTheIssuesReproduction:
    """The three dates from #195, run as they were reported."""

    def test_the_last_bar_is_eligible(self):
        rule = MarketCapRule(min_market_cap=50_000.0)

        assert rule.is_eligible(equity("AAA"), LAST_BAR, build_fetcher())

    def test_a_saturday_inside_the_data_is_eligible(self):
        """The defect: this used to be False, and so was every other name."""
        rule = MarketCapRule(min_market_cap=50_000.0)

        assert rule.is_eligible(equity("AAA"), SATURDAY_INSIDE_THE_DATA,
                                build_fetcher())

    def test_past_the_end_is_refused_rather_than_answered(self):
        rule = MarketCapRule(min_market_cap=50_000.0)

        with pytest.raises(CalculationError):
            rule.is_eligible(equity("AAA"), PAST_THE_END, build_fetcher())


class TestAClosedDayResolvesBackwards:
    """A gap inside the coverage is a shut market, not an absent price."""

    def test_a_weekend_matches_the_session_it_resolved_to(self):
        fetcher = build_fetcher()
        rule = MarketCapRule(min_market_cap=50_000.0)

        saturday = [asset.ticker for asset in universe()
                    if rule.is_eligible(asset, SATURDAY_INSIDE_THE_DATA, fetcher)]
        friday = [asset.ticker for asset in universe()
                  if rule.is_eligible(asset, THE_FRIDAY_BEFORE, fetcher)]

        assert saturday == friday

    def test_the_universe_does_not_empty_on_a_closed_day(self):
        """The reported harm: every name excluded, so nothing was left to weight."""
        result = select_with_provenance(universe(),
                                        [MarketCapRule(min_market_cap=50_000.0)],
                                        SATURDAY_INSIDE_THE_DATA,
                                        build_fetcher())

        assert result.survivor_ids == ["AAA"]

    def test_a_real_exclusion_still_excludes_on_a_closed_day(self):
        """Resolution must not turn the rule into a pass-through."""
        rule = MarketCapRule(min_market_cap=50_000.0)

        assert not rule.is_eligible(equity("ZZZ"), SATURDAY_INSIDE_THE_DATA,
                                    build_fetcher())

    def test_selection_and_weighting_resolve_to_the_same_session(self):
        """The two layers share one primitive, so they cannot disagree."""
        fetcher = build_fetcher()

        assert (fetcher.resolve_session(SATURDAY_INSIDE_THE_DATA)
                == THE_FRIDAY_BEFORE)


class TestPastTheLastBarRefuses:
    """A rule that cannot evaluate has not found the asset ineligible."""

    def test_the_message_names_the_date_asked_for_and_the_data_s_end(self):
        with pytest.raises(CalculationError) as raised:
            MarketCapRule(min_market_cap=50_000.0).is_eligible(
                equity("AAA"), PAST_THE_END, build_fetcher())

        message = str(raised.value)

        assert "2026-09-11" in message, "the requested date is not named"
        assert "2026-08-31" in message, "the data's actual end is not named"

    def test_the_refusal_names_the_rule_that_made_it(self):
        with pytest.raises(CalculationError, match="MarketCapRule"):
            MarketCapRule().is_eligible(equity("AAA"), PAST_THE_END,
                                        build_fetcher())

    def test_it_is_not_spelled_the_same_way_as_an_exclusion(self):
        """A False and a raise must stay distinguishable to the caller."""
        fetcher = build_fetcher()
        rule = MarketCapRule(min_market_cap=50_000.0)

        assert rule.is_eligible(equity("ZZZ"), LAST_BAR, fetcher) is False

        with pytest.raises(CalculationError):
            rule.is_eligible(equity("ZZZ"), PAST_THE_END, fetcher)

    def test_selection_surfaces_it_rather_than_emptying_the_universe(self):
        with pytest.raises(CalculationError):
            select_with_provenance(universe(),
                                   [MarketCapRule(min_market_cap=50_000.0)],
                                   PAST_THE_END,
                                   build_fetcher())


class TestTheSwallowedErrorsAreGone:
    """A rule that throws is not a rule that says no."""

    def test_a_broken_rule_propagates_rather_than_excluding(self):
        with pytest.raises(RuntimeError, match="the data source is down"):
            Boom().is_eligible(equity("AAA"), LAST_BAR, build_fetcher())

    def test_selection_propagates_it_too(self):
        """The same swallow lived one layer up, and would have undone this."""
        with pytest.raises(RuntimeError, match="the data source is down"):
            select_with_provenance(universe(), [Boom()], LAST_BAR,
                                   build_fetcher())

    def test_a_broken_rule_does_not_quietly_shrink_the_universe(self):
        """What used to happen: AAA excluded, the index published without it."""
        with pytest.raises(RuntimeError):
            select_with_provenance(universe(), [Boom()], LAST_BAR,
                                   build_fetcher())

    @pytest.mark.parametrize("rule_class", [MarketCapRule, LiquidityRule])
    def test_no_rule_catches_anything(self,
                                      rule_class):
        """Counted structurally: reintroducing a catch is a deliberate act.

        Nothing left in either body means "ineligible" by raising — the guards
        already answer False for a missing price, missing shares and a short
        window — so any handler here could only be hiding a fault.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(rule_class)))
        handlers = [node for node in ast.walk(tree)
                    if isinstance(node, ast.ExceptHandler)]

        assert handlers == [], f"{rule_class.__name__} catches an exception again"


class TestLiquidityRuleKeepsItsWindow:
    """It reads a lookback, so a closed day is spanned rather than landed on."""

    def test_a_weekend_is_not_a_special_case(self):
        rule = LiquidityRule(min_avg_daily_volume=1_000, lookback_days=10)
        fetcher = build_fetcher()

        assert rule.is_eligible(equity("AAA"), SATURDAY_INSIDE_THE_DATA, fetcher)
        assert rule.is_eligible(equity("AAA"), THE_FRIDAY_BEFORE, fetcher)

    def test_a_volume_floor_still_bites(self):
        rule = LiquidityRule(min_avg_daily_volume=VOLUME * 10, lookback_days=10)

        assert not rule.is_eligible(equity("AAA"), LAST_BAR, build_fetcher())

    def test_a_value_floor_still_bites(self):
        rule = LiquidityRule(min_avg_daily_value=1e15, lookback_days=10)

        assert not rule.is_eligible(equity("AAA"), LAST_BAR, build_fetcher())
