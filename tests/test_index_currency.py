# tests/test_index_currency.py
"""BN-188: a market cap is money, so a methodology has to say whose.

The fixture that would have caught the defect. `MarketCapWeighted` computed
``price x shares`` in whatever currency the name traded in and summed the
results, so a universe spanning currencies was weighted on incomparable
numbers — the yen name below carried 90.9% of an index in which it is worth
6.25%, a fifteen-fold error, and every other constituent's weight was wrong
with it.

It survived because nothing compared the two sides. `/data/reference`
converted its market-cap column and the weighting did not, so the table the
owner looked at was right about a name the index was weighting wrongly. The
tests here assert them against the *same* hand-computed number for that
reason, rather than each against itself.

The universe is the issue's: one dollar name and one yen name at 150, chosen
so the unconverted answer and the converted one are not near neighbours.
"""
import pandas as pd
import pytest

from beacon.asset.equity import Equity
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import CalculationError
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.context import IndexContext
from beacon.index.methodology import MarketCapRule, MarketCapWeighted
from beacon.server.reference import build_entries

START = "2024-01-02"
END = "2024-01-31"
DATES = pd.bdate_range(START, END)
AS_OF = pd.Timestamp(END)

# 150 yen to the dollar. Round, and far enough from one that no assertion here
# passes by accident.
JPY_PER_USD = 150.0

# USBIG: 100 USD x 1e9 = 100bn USD.
# JPSML: 1000 JPY x 1e9 = 1000bn JPY = 6.667bn USD.
SHARES = 1e9
PRICE = {"USBIG": 100.0, "JPSML": 1000.0}
CURRENCY = {"USBIG": "USD", "JPSML": "JPY"}

# 100 / (100 + 1000/150) — the weights the index should publish.
CONVERTED = {"USBIG": 0.9375, "JPSML": 0.0625}

# 100 / (100 + 1000) — the weights it did publish, kept as a named constant so
# the tests can assert the defect is gone rather than merely that some number
# looks plausible.
UNCONVERTED = {"USBIG": 1.0 / 11.0, "JPSML": 10.0 / 11.0}

USD_CONTEXT = IndexContext(currency="USD")


def build_fetcher(with_rates: bool = True,
                  free_float: dict[str, float] | None = None) -> DataFetcher:
    """The two-name, two-currency universe, with or without the JPYUSD pair."""
    rows: list[dict[str, object]] = []
    floats = free_float or dict.fromkeys(PRICE, 1.0)

    for date in DATES:
        for name, price in PRICE.items():
            rows.append({"IDENTIFIER": name, "DATE": date, "CLOSE": price,
                         "SHARES_OUTSTANDING": SHARES,
                         "FREE_FLOAT": floats[name]})

        if with_rates:
            rows.append({"IDENTIFIER": "JPYUSD", "DATE": date,
                         "RATE": 1.0 / JPY_PER_USD})

    reference = pd.DataFrame([
        {"IDENTIFIER": name, "DATE_FROM": "2020-01-01", "NAME": name,
         "CURRENCY": CURRENCY[name], "EXCHANGE": "XXXX"}
        for name in PRICE
    ])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def assets() -> list[Equity]:
    return [Equity(name=name, currency=CURRENCY[name], ticker=name,
                   exchange="XXXX")
            for name in PRICE]


def weights_by_ticker(scheme: MarketCapWeighted,
                      fetcher: DataFetcher,
                      context: IndexContext | None = USD_CONTEXT
                      ) -> dict[str, float]:
    computed = scheme.calculate_weights(assets(), AS_OF, fetcher, context)

    return {asset.asset_id: weight for asset, weight in computed.items()}


class TestTheWeightingConverts:
    """The defect itself."""

    def test_the_weights_are_the_converted_ones(self):
        weights = weights_by_ticker(MarketCapWeighted(), build_fetcher())

        for ticker, expected in CONVERTED.items():
            assert weights[ticker] == pytest.approx(expected, rel=1e-9)

    def test_the_yen_name_no_longer_dominates(self):
        """The shape of the bug: the smaller company held nine tenths."""
        weights = weights_by_ticker(MarketCapWeighted(), build_fetcher())

        assert weights["JPSML"] != pytest.approx(UNCONVERTED["JPSML"], rel=1e-6)
        assert weights["JPSML"] < weights["USBIG"]

    def test_it_converts_into_the_index_currency_not_a_fixed_one(self):
        """A yen index weights the same names by the same ratios.

        Weights are ratios, so the base cancels — what the index currency
        decides is which pairs have to exist, not what the answer is. Asserting
        it here keeps anyone from "fixing" this by hard-coding USD, which would
        leave a EUR index with dollars in its weights and euros on its levels.
        """
        rows = [{"IDENTIFIER": "USDJPY", "DATE": date, "RATE": JPY_PER_USD}
                for date in DATES]
        fetcher = build_fetcher(with_rates=False)
        fetcher.merge_market_data(pd.DataFrame(rows))

        weights = weights_by_ticker(MarketCapWeighted(), fetcher,
                                    IndexContext(currency="JPY"))

        for ticker, expected in CONVERTED.items():
            assert weights[ticker] == pytest.approx(expected, rel=1e-9)

    def test_free_float_is_applied_to_the_converted_cap(self):
        """Both adjustments land, and in a way that still sums to one."""
        floats = {"USBIG": 0.5, "JPSML": 1.0}
        fetcher = build_fetcher(free_float=floats)

        weights = weights_by_ticker(MarketCapWeighted(use_free_float=True),
                                    fetcher)

        caps = {name: PRICE[name] * SHARES * floats[name]
                * (1.0 if CURRENCY[name] == "USD" else 1.0 / JPY_PER_USD)
                for name in PRICE}
        total = sum(caps.values())

        for name, cap in caps.items():
            assert weights[name] == pytest.approx(cap / total, rel=1e-9)

    def test_it_drives_the_index_through_the_calculator(self):
        """End to end: the calculator supplies the context it never used to."""
        definition = IndexDefinition(index_id="FX",
                                     index_name="Two currencies",
                                     base_date=START,
                                     base_value=1000.0,
                                     currency="USD",
                                     eligibility_rules=[],
                                     weighting_scheme=MarketCapWeighted(),
                                     rebalancing_frequency="ANNUAL",
                                     calendar="XNYS",
                                     universe_identifiers=list(PRICE))

        result = IndexCalculator(definition, build_fetcher()).run(
            start_date=START, end_date=END)

        weights = result.weight_snapshots[min(result.weight_snapshots)]

        for ticker, expected in CONVERTED.items():
            assert weights[ticker] == pytest.approx(expected, rel=1e-6)


class TestAMissingRateRefuses:
    """BN-179's rule, applied to FX: no silent substitution of the local number."""

    def test_the_weighting_refuses(self):
        with pytest.raises(CalculationError) as excinfo:
            weights_by_ticker(MarketCapWeighted(),
                              build_fetcher(with_rates=False))

        assert "JPY/USD" in str(excinfo.value)

    def test_the_refusal_names_the_date_and_the_name(self):
        with pytest.raises(CalculationError) as excinfo:
            weights_by_ticker(MarketCapWeighted(),
                              build_fetcher(with_rates=False))

        message = str(excinfo.value)

        assert "JPSML" in message
        assert AS_OF.strftime("%Y-%m-%d") in message

    def test_the_rule_refuses(self):
        rule = MarketCapRule(min_market_cap=1.0)

        with pytest.raises(CalculationError, match="JPY/USD"):
            rule.is_eligible(assets()[1], AS_OF, build_fetcher(with_rates=False),
                             USD_CONTEXT)

    def test_a_multi_currency_universe_without_a_context_refuses(self):
        """Rather than quietly picking a base, which is the same fault again."""
        with pytest.raises(CalculationError, match="currencies"):
            weights_by_ticker(MarketCapWeighted(), build_fetcher(),
                              context=None)


class TestASingleCurrencyIndexIsUnmoved:
    """BN-188 must not restate a single-currency index by a single basis point."""

    @staticmethod
    def single_currency_fetcher() -> DataFetcher:
        """The same two names, both quoted in dollars, and no FX pair at all."""
        rows = [{"IDENTIFIER": name, "DATE": date, "CLOSE": price,
                 "SHARES_OUTSTANDING": SHARES}
                for name, price in PRICE.items()
                for date in DATES]
        reference = pd.DataFrame([
            {"IDENTIFIER": name, "DATE_FROM": "2020-01-01", "NAME": name,
             "CURRENCY": "USD", "EXCHANGE": "NYSE"}
            for name in PRICE
        ])

        return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                           ReferenceData.from_dataframe(reference))

    @staticmethod
    def dollar_assets() -> list[Equity]:
        return [Equity(name=name, currency="USD", ticker=name, exchange="NYSE")
                for name in PRICE]

    def test_the_weights_are_the_raw_cap_shares(self):
        scheme = MarketCapWeighted()
        computed = scheme.calculate_weights(self.dollar_assets(), AS_OF,
                                            self.single_currency_fetcher(),
                                            USD_CONTEXT)
        weights = {asset.asset_id: weight for asset, weight in computed.items()}

        for ticker, expected in UNCONVERTED.items():
            assert weights[ticker] == pytest.approx(expected, rel=1e-9)

    def test_a_universe_in_one_foreign_currency_needs_no_pair(self):
        """A common factor cancels out of a ratio.

        A yen-only universe in a dollar index weights identically whether or
        not a JPYUSD rate exists, so refusing for want of one would fail an
        index over a question that cannot change its answer.
        """
        yen = [Equity(name=name, currency="JPY", ticker=name, exchange="XTKS")
               for name in PRICE]
        computed = MarketCapWeighted().calculate_weights(
            yen, AS_OF, self.single_currency_fetcher(), USD_CONTEXT)
        weights = {asset.asset_id: weight for asset, weight in computed.items()}

        for ticker, expected in UNCONVERTED.items():
            assert weights[ticker] == pytest.approx(expected, rel=1e-9)

    def test_the_context_makes_no_difference(self):
        scheme = MarketCapWeighted()
        fetcher = self.single_currency_fetcher()

        with_context = scheme.calculate_weights(self.dollar_assets(), AS_OF,
                                                fetcher, USD_CONTEXT)
        without = scheme.calculate_weights(self.dollar_assets(), AS_OF, fetcher)

        assert ({asset.asset_id: w for asset, w in with_context.items()}
                == {asset.asset_id: w for asset, w in without.items()})


class TestTheRuleScreensOnConvertedCaps:
    """The bound says "in the index currency", and now the arithmetic does too."""

    def test_a_name_admitted_only_by_its_local_number_is_excluded(self):
        """JPSML's local cap reads 1000bn; it is worth 6.7bn."""
        rule = MarketCapRule(min_market_cap=10e9)
        fetcher = build_fetcher()
        usbig, jpsml = assets()

        assert rule.is_eligible(usbig, AS_OF, fetcher, USD_CONTEXT)
        assert not rule.is_eligible(jpsml, AS_OF, fetcher, USD_CONTEXT)

    def test_it_admitted_that_name_on_the_unconverted_number(self):
        """The defect, stated as a test: without a currency the bound is local."""
        rule = MarketCapRule(min_market_cap=10e9)

        assert rule.is_eligible(assets()[1], AS_OF, build_fetcher())

    def test_a_ceiling_no_longer_excludes_the_smaller_company(self):
        """The other direction: the yen name was too big for a 500bn ceiling."""
        rule = MarketCapRule(max_market_cap=500e9)

        assert rule.is_eligible(assets()[1], AS_OF, build_fetcher(), USD_CONTEXT)
        assert not rule.is_eligible(assets()[1], AS_OF, build_fetcher())


class TestTheDisplayAndTheWeightingAgree:
    """Half the reason this went unnoticed: the two sides never met."""

    def test_the_reported_caps_carry_the_same_ratio_as_the_weights(self):
        fetcher = build_fetcher()
        entries = build_entries(fetcher, list(PRICE), END, ["market_cap"])
        caps = {entry.identifier: entry.fields["market_cap"]
                for entry in entries}

        total = sum(caps.values())
        weights = weights_by_ticker(MarketCapWeighted(), fetcher)

        for ticker, cap in caps.items():
            assert weights[ticker] == pytest.approx(cap / total, rel=1e-9)

    def test_an_unconvertible_cap_is_reported_unknown_not_unconverted(self):
        """It used to fall back to a rate of 1.0 and label yen as dollars."""
        entries = build_entries(build_fetcher(with_rates=False), list(PRICE),
                                END, ["market_cap"])
        caps = {entry.identifier: entry.fields["market_cap"]
                for entry in entries}

        assert caps["USBIG"] == pytest.approx(PRICE["USBIG"] * SHARES)
        assert caps["JPSML"] is None

    def test_one_unconvertible_name_does_not_fail_the_batch(self):
        """A display endpoint renders the rest; only a methodology refuses."""
        entries = build_entries(build_fetcher(with_rates=False), list(PRICE),
                                END, ["market_cap"])

        assert [entry.found for entry in entries] == [True, True]


class TestOneRateLookup:
    """Three implementations disagreed; `fx_rate_on` is the only one left."""

    def test_a_currency_converts_into_itself_at_one(self):
        assert build_fetcher().fx_rate_on("USD", "USD", AS_OF) == 1.0

    def test_an_unknown_pair_is_none_rather_than_one(self):
        """The whole distinction: None is "cannot", 1.0 is "at parity"."""
        assert build_fetcher().fx_rate_on("CHF", "USD", AS_OF) is None

    def test_a_rate_is_carried_forward_over_a_gap(self):
        fetcher = build_fetcher()
        after_the_data = AS_OF + pd.Timedelta(days=30)

        assert fetcher.fx_rate_on("JPY", "USD", after_the_data) == pytest.approx(
            1.0 / JPY_PER_USD)

    def test_the_calculator_reads_the_same_lookup(self):
        """`rate_on` is the calculator's spelling of it, not a second one."""
        fetcher = build_fetcher()
        definition = IndexDefinition(index_id="FX",
                                     index_name="Two currencies",
                                     base_date=START,
                                     base_value=1000.0,
                                     currency="USD",
                                     eligibility_rules=[],
                                     weighting_scheme=MarketCapWeighted(),
                                     rebalancing_frequency="ANNUAL",
                                     calendar="XNYS",
                                     universe_identifiers=list(PRICE))
        calculator = IndexCalculator(definition, fetcher)

        assert (calculator.rate_on("JPY", "USD", AS_OF)
                == fetcher.fx_rate_on("JPY", "USD", AS_OF))

    def test_merging_market_data_does_not_leave_a_stale_rate(self):
        fetcher = build_fetcher()

        assert fetcher.fx_rate_on("JPY", "USD", AS_OF) == pytest.approx(
            1.0 / JPY_PER_USD)

        restated = pd.DataFrame([{"IDENTIFIER": "JPYUSD", "DATE": date,
                                  "RATE": 1.0 / 100.0}
                                 for date in DATES])
        fetcher.merge_market_data(restated)

        assert fetcher.fx_rate_on("JPY", "USD", AS_OF) == pytest.approx(0.01)
