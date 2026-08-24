# tests/test_expression_rules.py
"""BN-142: `ExpressionRule`.

The two tests that matter are the equivalence one — an expression must select
what the equivalent named rule selects, or the new surface is a second opinion
rather than a second spelling — and the look-ahead one, which is the only test
that tells a point-in-time screen apart from one that reads the latest value.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.asset.equity import Equity
from beacon.data.corporate_actions import CorporateActions
from beacon.data.features import FeatureData
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import InvalidRuleError
from beacon.expressions import data
from beacon.expressions.resolve import value_of
from beacon.index import ExpressionRule, MarketCapRule
from beacon.testing import dataset

REBALANCE = pd.Timestamp("2024-06-03")
NAMES = ["AAA", "BBB", "CCC", "DDD"]


def equity(identifier: str) -> Equity:
    return Equity(asset_id=identifier, ticker=identifier, name=identifier,
                  currency="USD", exchange="XNYS")


@pytest.fixture(scope="module")
def fetcher():
    return dataset.data_fetcher()


def with_features(frame: pd.DataFrame) -> DataFetcher:
    """The canonical dataset, carrying a feature table."""
    source = dataset.data_fetcher()

    return DataFetcher(source.market, source.reference,
                       source.corporate_actions,
                       FeatureData.from_dataframe(frame))


class TestItSelectsWhatTheNamedRuleSelects:
    """The acceptance case: a second spelling, not a second opinion."""

    @pytest.mark.parametrize("threshold", [0, 1e6, 1e9, 1e12, 1e15])
    def test_market_cap_agrees_with_market_cap_rule(self,
                                                    threshold,
                                                    fetcher):
        """Swept across thresholds rather than checked at one.

        A single threshold that happens to include everything or exclude
        everything would pass whatever the resolver computed — the two rules
        would agree by both saying yes to all four names. The sweep is what
        makes the agreement mean something.
        """
        expression = ExpressionRule.from_expression(
            data.market.market_cap > threshold)
        named = MarketCapRule(min_market_cap=threshold)

        for identifier in NAMES:
            asset = equity(identifier)

            assert (expression.is_eligible(asset, REBALANCE, fetcher)
                    == named.is_eligible(asset, REBALANCE, fetcher)), (
                f"{identifier} at {threshold:g}")

    def test_the_sweep_is_not_vacuous(self,
                                      fetcher):
        """Guards the test above: if every threshold selected the same names,
        the agreement would be trivially true and would stay true if the
        resolver broke."""
        selected = {
            threshold: sum(
                ExpressionRule.from_expression(data.market.market_cap
                                               > threshold)
                .is_eligible(equity(name), REBALANCE, fetcher)
                for name in NAMES)
            for threshold in (0, 1e9, 1e15)}

        assert len(set(selected.values())) > 1, (
            f"every threshold selects the same count: {selected}")


class TestItCannotLookAhead:
    """The test that distinguishes a point-in-time screen from a broken one."""

    def announced(self,
                  date: str) -> DataFetcher:
        """A store where the one feature value is published on `date`."""
        return with_features(pd.DataFrame([
            {"IDENTIFIER": name, "DATE": date, "TYPE": "fundamentals",
             "FIELD": "revenue", "VALUE": 5.0, "DETAIL": "FY24Q1"}
            for name in NAMES]))

    def selected(self,
                 fetcher: DataFetcher) -> set[str]:
        rule = ExpressionRule.from_expression(
            data.features.fundamentals.revenue > 1)

        return {name for name in NAMES
                if rule.is_eligible(equity(name), REBALANCE, fetcher)}

    def test_a_value_published_before_the_rebalance_is_seen(self):
        assert self.selected(self.announced("2024-05-15")) == set(NAMES)

    def test_a_value_published_after_it_is_invisible(self):
        """Q1 revenue announced in mid-July is not knowable at a June
        rebalance, however completely the quarter had ended."""
        assert self.selected(self.announced("2024-07-15")) == set()

    def test_moving_the_announcement_changes_the_constituents(self):
        """Stated as the issue states it: moving the announcement date past
        the rebalance changes the list at that rebalance and not before.

        A resolver reading the latest value regardless of date passes both
        halves of this individually and fails here.
        """
        before = self.selected(self.announced("2024-05-15"))
        after = self.selected(self.announced("2024-07-15"))

        assert before != after
        assert before and not after

    def test_a_later_rebalance_does_see_it(self):
        """The other half: the value is not lost, only not yet knowable."""
        rule = ExpressionRule.from_expression(
            data.features.fundamentals.revenue > 1)
        fetcher = self.announced("2024-07-15")

        assert rule.is_eligible(equity("AAA"), pd.Timestamp("2024-08-01"),
                                fetcher)


class TestMissingCoverage:
    """A stated behaviour, and distinct from a legitimate zero."""

    def rule_over(self,
                  on_missing: str) -> ExpressionRule:
        return ExpressionRule.from_expression(
            data.features.fundamentals.revenue > 1, on_missing=on_missing)

    def test_a_name_with_no_value_is_excluded_by_default(self):
        """A screen that silently admits uncovered names is not the screen it
        claims to be."""
        covered = with_features(pd.DataFrame([
            {"IDENTIFIER": "AAA", "DATE": "2024-01-15",
             "TYPE": "fundamentals", "FIELD": "revenue", "VALUE": 5.0,
             "DETAIL": None}]))

        assert self.rule_over("exclude").is_eligible(equity("AAA"), REBALANCE,
                                                     covered)
        assert not self.rule_over("exclude").is_eligible(equity("BBB"),
                                                         REBALANCE, covered)

    def test_including_is_available(self):
        covered = with_features(pd.DataFrame([
            {"IDENTIFIER": "AAA", "DATE": "2024-01-15",
             "TYPE": "fundamentals", "FIELD": "revenue", "VALUE": 5.0,
             "DETAIL": None}]))

        assert self.rule_over("include").is_eligible(equity("BBB"), REBALANCE,
                                                     covered)

    def test_zero_is_not_missing(self):
        """Zero fails a `> 1` test honestly; missing has nothing to compare.
        Collapsing them would make `on_missing` decide the answer for a name
        whose value is perfectly well known."""
        zeroed = with_features(pd.DataFrame([
            {"IDENTIFIER": "AAA", "DATE": "2024-01-15",
             "TYPE": "fundamentals", "FIELD": "revenue", "VALUE": 0.0,
             "DETAIL": None}]))

        assert not self.rule_over("include").is_eligible(equity("AAA"),
                                                         REBALANCE, zeroed)

    def test_an_unknown_on_missing_is_refused(self):
        with pytest.raises(InvalidRuleError):
            ExpressionRule.from_expression(data.market.close > 1,
                                           on_missing="maybe")


class TestTheStoredForm:
    """Expression → document → reload → same answer."""

    def test_it_stores_its_tree_in_params(self):
        rule = ExpressionRule.from_expression(data.market.close > 1)

        assert rule.expression["node"] == "comparison"

    def test_a_reloaded_rule_selects_identically(self,
                                                 fetcher):
        """The round trip the issue asks for. A definition saved and reloaded
        must screen the same, or a backtest is not reproducible."""
        original = ExpressionRule.from_expression(
            (data.market.market_cap > 1e9)
            & ~(data.reference.sector == "Energy"))
        reloaded = ExpressionRule(original.expression)

        for name in NAMES:
            asset = equity(name)

            assert (original.is_eligible(asset, REBALANCE, fetcher)
                    == reloaded.is_eligible(asset, REBALANCE, fetcher))

    def test_a_malformed_tree_is_refused_at_construction(self):
        """A malformed tree is a fact about the rule. Finding out at the first
        rebalance means a run that dies partway through with thousands of
        names already priced."""
        with pytest.raises(InvalidRuleError):
            ExpressionRule({"node": "sometimes"})

    def test_it_is_in_the_rule_catalogue(self):
        """A rule the editor never offers may as well not exist, and the
        failure mode of forgetting to register is silent."""
        from beacon.catalogue import SELECTION, entries

        assert "ExpressionRule" in {entry.name for entry in entries(SELECTION)}


class TestComposition:
    """Whole screens, not single comparisons."""

    def test_and_narrows(self,
                         fetcher):
        wide = ExpressionRule.from_expression(data.market.close > 0)
        narrow = ExpressionRule.from_expression(
            (data.market.close > 0) & (data.market.close > 1e9))

        assert wide.is_eligible(equity("AAA"), REBALANCE, fetcher)
        assert not narrow.is_eligible(equity("AAA"), REBALANCE, fetcher)

    def test_or_widens(self,
                       fetcher):
        rule = ExpressionRule.from_expression(
            (data.market.close > 1e9) | (data.market.close > 0))

        assert rule.is_eligible(equity("AAA"), REBALANCE, fetcher)

    def test_not_inverts(self,
                         fetcher):
        rule = ExpressionRule.from_expression(~(data.market.close > 0))

        assert not rule.is_eligible(equity("AAA"), REBALANCE, fetcher)

    def test_between_and_is_in(self,
                               fetcher):
        sector = value_of(data.reference.sector, "AAA", REBALANCE, fetcher)
        rule = ExpressionRule.from_expression(
            data.reference.sector.is_in([sector, "Nowhere"]))

        assert rule.is_eligible(equity("AAA"), REBALANCE, fetcher)


class TestResolution:
    """What `value_of` reads, per namespace."""

    def test_a_market_column_is_as_of_the_date(self,
                                               fetcher):
        early = value_of(data.market.close, "AAA", pd.Timestamp("2024-02-01"),
                         fetcher)
        late = value_of(data.market.close, "AAA", REBALANCE, fetcher)

        assert early != late

    def test_a_non_trading_date_falls_back(self,
                                           fetcher):
        """A rebalance can land on a day an instrument did not trade, and
        dropping it for that reason has nothing to do with the screen."""
        saturday = pd.Timestamp("2024-06-01")

        assert value_of(data.market.close, "AAA", saturday, fetcher) is not None

    def test_an_unknown_instrument_has_no_value(self,
                                                fetcher):
        assert value_of(data.market.close, "NOSUCH", REBALANCE, fetcher) is None

    def test_a_derived_field_is_computed(self,
                                         fetcher):
        cap = value_of(data.market.market_cap, "AAA", REBALANCE, fetcher)
        close = value_of(data.market.close, "AAA", REBALANCE, fetcher)
        shares = value_of(data.market.shares_outstanding, "AAA", REBALANCE,
                          fetcher)

        assert cap == pytest.approx(float(close) * float(shares))

    def test_free_float_cap_is_the_scaled_one(self,
                                              fetcher):
        cap = value_of(data.market.market_cap, "AAA", REBALANCE, fetcher)
        floated = value_of(data.market.free_float_market_cap, "AAA", REBALANCE,
                           fetcher)

        assert floated is None or floated <= cap

    def test_adv_is_a_trailing_mean(self,
                                    fetcher):
        adv = value_of(data.market.adv_3m, "AAA", REBALANCE, fetcher)

        assert adv is not None and adv > 0

    def test_an_action_field_has_no_scalar_value(self,
                                                 fetcher):
        """Corporate actions are events, not attributes: an instrument has a
        history of them rather than one value on a date, so there is nothing
        for a scalar comparison to read."""
        assert value_of(data.actions.value, "AAA", REBALANCE, fetcher) is None


class TestCurrencyConversion:
    """A cap comparison must not be a currency comparison."""

    def test_a_foreign_cap_is_converted(self):
        """Since BN-128 one universe spans seven currencies. Comparing raw
        local values ranks a yen cap above a dollar one on magnitude alone,
        so `market_cap > 1e9` would select on currency as much as on size.
        """
        dates = pd.date_range("2024-05-01", "2024-06-03", freq="B")

        market = pd.DataFrame({
            "IDENTIFIER": "JPY1", "DATE": dates, "CLOSE": 100.0,
            "SHARES_OUTSTANDING": 1_000_000.0, "VOLUME": 1_000.0})
        rates = pd.DataFrame({
            "IDENTIFIER": "JPYUSD", "DATE": dates, "CLOSE": 0.0064})
        reference = pd.DataFrame([
            {"IDENTIFIER": "JPY1", "DATE_FROM": "2000-01-01",
             "NAME": "JPY One", "SECTOR": "Industrials",
             "CURRENCY": "JPY", "EXCHANGE": "XTKS"}])

        from beacon.data.base import MarketData, ReferenceData

        fetcher = DataFetcher(
            MarketData.from_dataframe(pd.concat([market, rates],
                                                ignore_index=True)),
            ReferenceData.from_dataframe(reference),
            CorporateActions.empty())

        cap = value_of(data.market.market_cap, "JPY1", REBALANCE, fetcher)
        unconverted = 100.0 * 1_000_000.0

        assert cap == pytest.approx(unconverted * 0.0064)
        assert not np.isclose(cap, unconverted)
