# tests/test_feature_rules.py
"""BN-136: index rules that screen on features.

The rule is where the point-in-time table either pays off or is quietly
bypassed. Reading the feature frame directly rather than through
`fetch_feature` would put look-ahead straight back in, and the backtest would
look *better* for it — which is why the test that matters here is not "does
the screen work" but "does moving a publication date change the answer".
"""

import pandas as pd
import pytest

from beacon.data.features import FeatureData
from beacon.data.fetcher import DataFetcher
from beacon.index import FeatureRule
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.testing import dataset

START = "2024-01-02"
END = "2024-09-30"


def features(*records) -> FeatureData:
    """A feature table from partial records."""
    base = {"IDENTIFIER": "AAA", "DATE": "2024-05-15",
            "TYPE": "fundamentals", "FIELD": "revenue", "VALUE": 1000.0,
            "DETAIL": None}

    return FeatureData.from_dataframe(
        pd.DataFrame([{**base, **record} for record in records]))


def fetcher_with(table: FeatureData) -> DataFetcher:
    source = dataset.data_fetcher()

    return DataFetcher(source.market, source.reference,
                       source.corporate_actions, table)


class TestScreening:
    """The ordinary behaviour."""

    @pytest.mark.parametrize("comparison,threshold,expected", [
        ("gt", 500.0, True),
        ("gt", 5000.0, False),
        ("lt", 5000.0, True),
        ("ge", 1000.0, True),
        ("le", 1000.0, True),
        ("eq", 1000.0, True),
        ("ne", 1000.0, False),
    ])
    def test_the_comparisons(self,
                             comparison,
                             threshold,
                             expected):
        rule = FeatureRule("revenue", comparison, threshold)
        provider = fetcher_with(features({}))
        asset = next(a for a in [_equity("AAA")])

        assert rule.is_eligible(asset, pd.Timestamp("2024-06-01"),
                                provider) is expected

    def test_a_type_selects_between_two_datasets(self):
        """`revenue` from a vendor and from a user's own model are different
        series, and a screen has to be able to say which."""
        provider = fetcher_with(features(
            {"TYPE": "fundamentals", "VALUE": 1000.0},
            {"TYPE": "derived", "VALUE": 9999.0}))
        asset = _equity("AAA")
        date = pd.Timestamp("2024-06-01")

        vendor = FeatureRule("revenue", "gt", 5000.0,
                             feature_type="fundamentals")
        own = FeatureRule("revenue", "gt", 5000.0, feature_type="derived")

        assert vendor.is_eligible(asset, date, provider) is False
        assert own.is_eligible(asset, date, provider) is True


class TestItCannotLookAhead:
    """The reason this issue exists."""

    def test_a_value_published_later_is_invisible(self):
        """Standing on 1 April, revenue announced on 15 May has not happened.
        A rule that saw it would select on information nobody had."""
        provider = fetcher_with(features({"DATE": "2024-05-15",
                                          "VALUE": 1000.0}))
        rule = FeatureRule("revenue", "gt", 500.0)

        assert rule.is_eligible(_equity("AAA"), pd.Timestamp("2024-04-01"),
                                provider) is False
        assert rule.is_eligible(_equity("AAA"), pd.Timestamp("2024-05-15"),
                                provider) is True

    def test_moving_the_announcement_changes_the_constituents(self):
        """The acceptance test of this issue, and the only one that
        distinguishes a point-in-time screen from a look-ahead one.

        The same value, the same threshold, the same rebalance — published
        either side of it. A rule reading the table directly would select the
        name in both runs and every other test here would still pass.
        """
        rebalance = pd.Timestamp("2024-06-03")
        rule = FeatureRule("revenue", "gt", 500.0)

        before = fetcher_with(features({"DATE": "2024-05-15"}))
        after = fetcher_with(features({"DATE": "2024-07-15"}))

        assert rule.is_eligible(_equity("AAA"), rebalance, before) is True
        assert rule.is_eligible(_equity("AAA"), rebalance, after) is False

    def test_a_restatement_does_not_leak_backwards(self):
        """Standing in June, the May figure is what was believed. The August
        revision is the future."""
        provider = fetcher_with(features(
            {"DATE": "2024-05-15", "VALUE": 1000.0},
            {"DATE": "2024-08-15", "VALUE": 9999.0}))
        rule = FeatureRule("revenue", "gt", 5000.0)

        assert rule.is_eligible(_equity("AAA"), pd.Timestamp("2024-06-01"),
                                provider) is False
        assert rule.is_eligible(_equity("AAA"), pd.Timestamp("2024-08-15"),
                                provider) is True


class TestMissingCoverage:
    """A decision, not an accident."""

    def test_an_uncovered_name_is_excluded_by_default(self):
        """A screen for "revenue above a billion" that silently admitted every
        company the dataset has never heard of would be the opposite of what
        it says."""
        provider = fetcher_with(features({"IDENTIFIER": "AAA"}))
        rule = FeatureRule("revenue", "gt", 500.0)

        assert rule.is_eligible(_equity("BBB"), pd.Timestamp("2024-06-01"),
                                provider) is False

    def test_it_can_be_included_instead(self):
        """Excluding is wrong too when coverage is patchy — the universe
        shrinks to the names the vendor happened to cover. So it is a
        parameter, and the default is the one whose failure is visible."""
        provider = fetcher_with(features({"IDENTIFIER": "AAA"}))
        rule = FeatureRule("revenue", "gt", 500.0, on_missing="include")

        assert rule.is_eligible(_equity("BBB"), pd.Timestamp("2024-06-01"),
                                provider) is True

    def test_missing_is_not_the_same_as_zero(self):
        """Zero fails a `> 0` test honestly. Missing has no value to compare
        at all, and conflating them would make a legitimately zero-revenue
        company indistinguishable from one nobody has data for."""
        provider = fetcher_with(features({"IDENTIFIER": "AAA",
                                          "VALUE": 0.0}))
        rule = FeatureRule("revenue", "gt", -1.0, on_missing="include")

        # AAA has a real zero and passes on its merits; BBB has nothing and
        # passes only because the rule was told to include the uncovered.
        assert rule.is_eligible(_equity("AAA"), pd.Timestamp("2024-06-01"),
                                provider) is True

        strict = FeatureRule("revenue", "gt", -1.0, on_missing="exclude")

        assert strict.is_eligible(_equity("AAA"), pd.Timestamp("2024-06-01"),
                                  provider) is True
        assert strict.is_eligible(_equity("BBB"), pd.Timestamp("2024-06-01"),
                                  provider) is False


class TestBadConfiguration:
    """Refused at construction, so a pipeline cannot carry a rule that will
    fail halfway through a run."""

    def test_an_unknown_comparison_is_refused(self):
        from beacon.exceptions import InvalidRuleError

        with pytest.raises(InvalidRuleError, match="comparison"):
            FeatureRule("revenue", comparison="approximately")

    def test_an_unknown_missing_policy_is_refused(self):
        from beacon.exceptions import InvalidRuleError

        with pytest.raises(InvalidRuleError, match="on_missing"):
            FeatureRule("revenue", on_missing="maybe")

    def test_an_empty_field_is_refused(self):
        from beacon.exceptions import InvalidRuleError

        with pytest.raises(InvalidRuleError, match="field"):
            FeatureRule("")


class TestItIsInTheCatalogue:
    """So the client can offer it without a code change."""

    def test_it_is_registered(self):
        from beacon.catalogue import SELECTION, registered_names

        assert "FeatureRule" in registered_names(SELECTION)


class TestEndToEnd:
    """Through a real index calculation."""

    def test_an_index_screens_on_a_feature(self):
        provider = fetcher_with(features(
            {"IDENTIFIER": "AAA", "VALUE": 1000.0, "DATE": "2024-01-02"},
            {"IDENTIFIER": "BBB", "VALUE": 10.0, "DATE": "2024-01-02"}))

        definition = IndexDefinition(
            index_id="FEAT", index_name="FEAT", base_date=START,
            base_value=1000.0, currency="USD",
            eligibility_rules=[FeatureRule("revenue", "gt", 500.0)],
            weighting_scheme=EqualWeighted(),
            rebalancing_frequency="QUARTERLY",
            universe_identifiers=["AAA", "BBB", "CCC"])

        result = IndexCalculator(definition, provider).run(start_date=START,
                                                           end_date=END)
        latest = result.constituent_snapshots[max(result.constituent_snapshots)]

        assert "AAA" in latest
        assert "BBB" not in latest, "a name below the threshold was selected"
        assert "CCC" not in latest, "an uncovered name was selected"


def _equity(identifier: str):
    """One asset, as the calculator builds them."""
    from beacon.asset.equity import Equity

    return Equity(name=identifier, currency="USD", ticker=identifier,
                  exchange="XNAS", asset_id=identifier)
