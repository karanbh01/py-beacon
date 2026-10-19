# tests/test_stage_descriptions.py
"""BN-328: settings formatted as a reader reads them, and the sentences."""
import pytest

from beacon.catalogue import BPS, COUNT, DAYS, FRACTION, MONEY
from beacon.server import stage_descriptions as describe


class TestFormatting:

    @pytest.mark.parametrize(("value", "unit", "text"), [
        (50e9, MONEY, "50,000,000,000 USD"), (5.5, MONEY, "5.50 USD"), (0.3, FRACTION, "30%"),
        (0.025, FRACTION, "2.5%"), (12, BPS, "12 bps"), (60, DAYS, "60 days"), (1, DAYS, "1 day"),
        (100000, COUNT, "100,000"), (True, None, "Yes"), (["A", "B", "C"], None, "A, B and C")])
    def test_a_value_reads_with_its_unit(self,
                                         value,
                                         unit,
                                         text):
        assert describe.formatted(value, unit, "USD") == text


class TestSentences:

    def test_a_market_cap_rule_with_both_bounds(self):
        about = describe.rule("MarketCapRule", {"min_market_cap": 1e9, "max_market_cap": 5e9},
                              "EUR")

        assert about.description == ("Keeps names whose market cap is between "
                                     "1,000,000,000 EUR and 5,000,000,000 EUR.")

    def test_a_screen_with_an_exit_says_a_held_name_stays(self):
        about = describe.screen("MinimumPriceScreen", {"min_price": 5, "exit": 4}, "USD")

        assert about.description == ("Drops names priced below 5 USD, though a name already "
                                     "held stays until it falls below 4 USD.")

    def test_a_false_default_is_not_listed(self):
        params = describe.screen("MinimumPriceScreen", {"min_price": 5}, "USD").params

        assert params == [{"label": "Minimum price", "value": "5 USD"}]

    def test_an_exclusion_lists_its_names_and_sectors(self):
        about = describe.screen("ExclusionScreen", {"identifiers": ["A", "B"],
                                                    "sectors": ["Tobacco"]}, "USD")

        assert about.description == "Drops A and B, and any name in Tobacco."

    def test_a_sampled_replication_says_how_many(self):
        about = describe.strategy({"type": "index_tracking",
                                   "replication": {"type": "SampledReplication",
                                                   "params": {"holdings": 50}}}, "USD")

        assert about.description.startswith("Holds the largest names in each sector")
        assert "50 names in all" in about.description

    def test_no_sentence_names_a_class(self):
        sentences = [describe.rule("LiquidityRule", {"min_avg_daily_volume": 1e5}, "USD"),
                     describe.screen("LiquidityScreen", {"min_traded_value": 1e6}, "USD"),
                     describe.capacity({"caps": [{"type": "OwnershipCap",
                                                  "params": {"max_share": 0.1}}]}, "USD")]

        for about in sentences:
            assert about.type not in about.description and "—" not in about.description
