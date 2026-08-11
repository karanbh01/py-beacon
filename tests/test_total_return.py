# tests/test_total_return.py
"""BN-125: total return and net total return.

A price index drops by the whole distribution on an ex-date; a total-return
index reinvests it. The tests that carry the issue are the two that would catch
the ways this goes wrong quietly:

* an index whose universe pays nothing must produce **identical** levels under
  all three return types — anything else means the machinery is doing something
  on days it should not touch
* a hand-computed two-name, two-dividend case must reconcile exactly, because
  every other test here checks a direction rather than a number, and a
  construction that is wrong by a constant factor satisfies all of them
"""
import pandas as pd
import pytest

from beacon.asset.base import Asset
from beacon.data.base import MarketData, ReferenceData
from beacon.data.corporate_actions import CorporateActions
from beacon.data.fetcher import DataFetcher
from beacon.index.calculation import IndexCalculator
from beacon.index.calculation.total_return import (
    NET_TOTAL_RETURN,
    PRICE,
    TOTAL_RETURN,
    TotalReturnMixin,
    withholding_for,
)
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted

START = "2025-01-01"
END = "2025-03-31"
DATES = pd.bdate_range(START, END)


def build_fetcher(actions: list[dict] | None = None,
                  prices: dict[str, float] | None = None) -> DataFetcher:
    """Two names on flat prices, with an optional action history.

    Flat prices on purpose: with no price movement, every difference between
    the three return types is the distribution and nothing else.
    """
    levels = prices or {"AAA": 100.0, "BBB": 50.0}

    market = pd.DataFrame([
        {"IDENTIFIER": name, "DATE": date, "CLOSE": price,
         "SHARES_OUTSTANDING": 1_000.0}
        for name, price in levels.items()
        for date in DATES])

    # Reference data is not optional here: `_get_universe` resolves names
    # through it, and a fetcher without it yields an empty index whose level
    # sits at the base value forever — which looks exactly like a total-return
    # calculation that does nothing.
    reference = ReferenceData.from_dataframe(pd.DataFrame([
        {"IDENTIFIER": name, "DATE_FROM": "2020-01-01", "NAME": name,
         "CURRENCY": "USD", "EXCHANGE": "XNYS"}
        for name in levels]))

    history = None
    if actions:
        history = CorporateActions.from_dataframe(pd.DataFrame(actions))

    return DataFetcher(MarketData.from_dataframe(market), reference, history)


def run(fetcher: DataFetcher,
        return_type: str = PRICE,
        rate: float = 0.0,
        universe: list[str] | None = None) -> pd.Series:
    """Calculate an equal-weighted index and return its levels."""
    definition = IndexDefinition(
        index_id="IDX", index_name="Index", base_date=START, base_value=1000.0,
        currency="USD", eligibility_rules=[], weighting_scheme=EqualWeighted(),
        rebalancing_frequency="ANNUAL",
        universe_identifiers=universe or ["AAA", "BBB"],
        return_type=return_type, withholding_tax_rate=rate)

    return IndexCalculator(definition, fetcher).run(
        start_date=START, end_date=END).index_levels


class TestNothingChangesWithoutDistributions:
    """The machinery must be inert when there is nothing to reinvest."""

    def test_all_three_types_agree_when_nothing_pays(self):
        """Identical, not approximately equal. A construction that touched the
        divisor on a day with no distribution would show up here and nowhere
        else."""
        fetcher = build_fetcher()

        price = run(fetcher, PRICE)
        gross = run(fetcher, TOTAL_RETURN)
        net = run(fetcher, NET_TOTAL_RETURN, 0.30)

        pd.testing.assert_series_equal(price, gross)
        pd.testing.assert_series_equal(price, net)

    def test_a_history_of_only_splits_changes_nothing(self):
        """A split is a ratio action. Reinvesting its ratio as cash would
        multiply the index by two."""
        fetcher = build_fetcher([
            {"IDENTIFIER": "AAA", "EX_DATE": DATES[10], "TYPE": "SPLIT",
             "VALUE": 2.0}])

        pd.testing.assert_series_equal(run(fetcher, PRICE),
                                       run(fetcher, TOTAL_RETURN))

    def test_a_structural_action_changes_nothing(self):
        """A spin-off carries no directly aggregable value."""
        fetcher = build_fetcher([
            {"IDENTIFIER": "AAA", "EX_DATE": DATES[10], "TYPE": "SPIN_OFF",
             "VALUE": 0.4}])

        pd.testing.assert_series_equal(run(fetcher, PRICE),
                                       run(fetcher, TOTAL_RETURN))

    def test_a_price_index_reads_no_action_history(self):
        """It must not merely ignore the result — it should not look. A price
        index built before BN-125 produced these levels and must still."""
        with_actions = build_fetcher([
            {"IDENTIFIER": "AAA", "EX_DATE": DATES[10], "TYPE": "DIVIDEND",
             "VALUE": 5.0}])

        pd.testing.assert_series_equal(run(with_actions, PRICE),
                                       run(build_fetcher(), PRICE))


class TestOrdering:
    """Directions that must hold on any data."""

    @pytest.fixture
    def paying(self) -> DataFetcher:
        return build_fetcher([
            {"IDENTIFIER": "AAA", "EX_DATE": DATES[10], "TYPE": "DIVIDEND",
             "VALUE": 2.0},
            {"IDENTIFIER": "BBB", "EX_DATE": DATES[30], "TYPE": "DIVIDEND",
             "VALUE": 1.0}])

    def test_total_return_is_never_below_price(self, paying):
        price, gross = run(paying, PRICE), run(paying, TOTAL_RETURN)

        assert (gross >= price - 1e-9).all()

    def test_total_return_ends_above_price(self, paying):
        assert run(paying, TOTAL_RETURN).iloc[-1] > run(paying, PRICE).iloc[-1]

    @pytest.mark.parametrize("rate", [0.15, 0.30, 0.50])
    def test_net_sits_between_price_and_gross(self, paying, rate):
        price = run(paying, PRICE).iloc[-1]
        net = run(paying, NET_TOTAL_RETURN, rate).iloc[-1]
        gross = run(paying, TOTAL_RETURN).iloc[-1]

        assert price < net < gross

    def test_a_higher_withholding_rate_gives_a_lower_level(self, paying):
        low = run(paying, NET_TOTAL_RETURN, 0.15).iloc[-1]
        high = run(paying, NET_TOTAL_RETURN, 0.50).iloc[-1]

        assert high < low

    def test_full_withholding_still_differs_from_price(self, paying):
        """At a rate approaching 1 the net index approaches the price index,
        but the rate is capped below 1 so they never coincide by construction."""
        net = run(paying, NET_TOTAL_RETURN, 0.99).iloc[-1]
        price = run(paying, PRICE).iloc[-1]

        assert net > price
        assert net == pytest.approx(price, rel=1e-3)


class TestHandComputed:
    """One case worked out by hand, reconciled exactly.

    Every other test here checks a direction. A construction wrong by a
    constant factor passes all of them and fails this one.
    """

    def test_a_single_dividend_reconciles(self):
        """Two names, equal-weighted, flat prices, one dividend.

        The index starts at 1000 holding an equal-weighted basket. AAA pays 2.0
        per share on day 10. Because prices are flat and weights equal, the
        distribution is worth exactly half of (2.0 / 100.0) of the index — the
        yield on the paying half — so the level steps from 1000 to
        1000 x (1 + 0.5 x 0.02) = 1010 and stays there.
        """
        fetcher = build_fetcher([
            {"IDENTIFIER": "AAA", "EX_DATE": DATES[10], "TYPE": "DIVIDEND",
             "VALUE": 2.0}])

        levels = run(fetcher, TOTAL_RETURN)

        assert levels.iloc[9] == pytest.approx(1000.0, abs=1e-9)
        assert levels.iloc[10] == pytest.approx(1010.0, abs=1e-6)
        assert levels.iloc[-1] == pytest.approx(1010.0, abs=1e-6)

    def test_two_dividends_compound(self):
        """AAA pays 2.0 on 100.0 (a 2% yield on half the index), then BBB pays
        1.0 on 50.0 (also 2% on half). The two compound rather than add:
        1000 x 1.01 x 1.01 = 1020.10, not 1020.
        """
        fetcher = build_fetcher([
            {"IDENTIFIER": "AAA", "EX_DATE": DATES[10], "TYPE": "DIVIDEND",
             "VALUE": 2.0},
            {"IDENTIFIER": "BBB", "EX_DATE": DATES[30], "TYPE": "DIVIDEND",
             "VALUE": 1.0}])

        levels = run(fetcher, TOTAL_RETURN)

        assert levels.iloc[10] == pytest.approx(1010.0, abs=1e-6)
        assert levels.iloc[30] == pytest.approx(1020.10, abs=1e-6)

    def test_withholding_scales_the_step(self):
        """A 25% rate keeps three quarters of the step: 1000 -> 1007.50."""
        fetcher = build_fetcher([
            {"IDENTIFIER": "AAA", "EX_DATE": DATES[10], "TYPE": "DIVIDEND",
             "VALUE": 2.0}])

        levels = run(fetcher, NET_TOTAL_RETURN, 0.25)

        assert levels.iloc[10] == pytest.approx(1007.50, abs=1e-6)

    def test_two_names_paying_on_one_date_both_count(self):
        fetcher = build_fetcher([
            {"IDENTIFIER": "AAA", "EX_DATE": DATES[10], "TYPE": "DIVIDEND",
             "VALUE": 2.0},
            {"IDENTIFIER": "BBB", "EX_DATE": DATES[10], "TYPE": "DIVIDEND",
             "VALUE": 1.0}])

        levels = run(fetcher, TOTAL_RETURN)

        assert levels.iloc[10] == pytest.approx(1020.0, abs=1e-6)

    def test_two_actions_on_one_name_and_date_are_summed(self):
        """An ordinary and a special dividend on the same ex-date. Summed, not
        replaced — a dict assignment would keep only the second."""
        fetcher = build_fetcher([
            {"IDENTIFIER": "AAA", "EX_DATE": DATES[10], "TYPE": "DIVIDEND",
             "VALUE": 1.0},
            {"IDENTIFIER": "AAA", "EX_DATE": DATES[10],
             "TYPE": "SPECIAL_DIVIDEND", "VALUE": 1.0}])

        levels = run(fetcher, TOTAL_RETURN)

        assert levels.iloc[10] == pytest.approx(1010.0, abs=1e-6)


class TestMechanics:
    """The pieces, independently of a run."""

    def test_reinvest_offsets_the_drop(self):
        divisor = TotalReturnMixin.reinvest(2.0, 100.0, 10.0)

        assert 100.0 / divisor == pytest.approx(110.0 / 2.0)

    def test_reinvest_is_a_no_op_without_a_distribution(self):
        assert TotalReturnMixin.reinvest(2.0, 100.0, 0.0) == 2.0

    def test_reinvest_refuses_a_worthless_aggregate(self):
        """An index with no value cannot reinvest into itself, and scaling by
        zero would destroy the divisor rather than adjust it."""
        assert TotalReturnMixin.reinvest(2.0, 0.0, 10.0) == 2.0

    def test_withholding_applies_only_to_net(self):
        assert withholding_for(NET_TOTAL_RETURN, 0.3) == 0.3
        assert withholding_for(TOTAL_RETURN, 0.3) == 0.0
        assert withholding_for(PRICE, 0.3) == 0.0

    def test_no_holdings_receive_nothing(self):
        assert TotalReturnMixin.distribution_received({}, {"AAA": 1.0}) == 0.0

    def test_no_distribution_is_nothing(self):
        assert TotalReturnMixin.distribution_received({"a": 1.0}, {}) == 0.0

    def test_a_foreign_dividend_is_converted_before_it_is_reinvested(self):
        """The bug the global universe exposed, pinned.

        A dividend is quoted in the paying company's currency; the aggregate
        it is reinvested into is in the index's. Summing the two without
        converting counted a five-yen dividend as five dollars, and a
        twelve-name index came out yielding 37% a year. Against a
        single-currency universe every rate is 1.0, so nothing could catch it.
        """
        units = {Asset(name="JPY Co", currency="JPY", asset_id="JPYCO",
                       asset_type="EQUITY"): 100.0}
        per_share = {"JPYCO": 5.0}

        unconverted = TotalReturnMixin.distribution_received(units, per_share)
        converted = TotalReturnMixin.distribution_received(
            units, per_share, rates={"JPYCO": 1.0 / 157.0})

        assert unconverted == pytest.approx(500.0)
        assert converted == pytest.approx(500.0 / 157.0)
        assert converted < unconverted / 100

    def test_a_name_already_in_index_currency_needs_no_rate(self):
        """A missing entry means "no conversion needed", not "unknown". Every
        domestic name would otherwise need a redundant 1.0 in the mapping."""
        units = {Asset(name="USD Co", currency="USD", asset_id="USDCO",
                       asset_type="EQUITY"): 10.0}

        assert TotalReturnMixin.distribution_received(
            units, {"USDCO": 2.0}, rates={}) == pytest.approx(20.0)


class TestDefinitionValidation:
    """What the library refuses."""

    def test_an_unknown_return_type_is_refused(self):
        with pytest.raises(ValueError, match="Unsupported return type"):
            IndexDefinition(
                index_id="I", index_name="I", base_date=START, base_value=1000.0,
                currency="USD", eligibility_rules=[],
                weighting_scheme=EqualWeighted(),
                rebalancing_frequency="ANNUAL", return_type="GROSS_OF_FEES")

    @pytest.mark.parametrize("rate", [-0.1, 1.0, 1.5])
    def test_an_impossible_withholding_rate_is_refused(self, rate):
        with pytest.raises(ValueError, match="withholding_tax_rate"):
            IndexDefinition(
                index_id="I", index_name="I", base_date=START, base_value=1000.0,
                currency="USD", eligibility_rules=[],
                weighting_scheme=EqualWeighted(),
                rebalancing_frequency="ANNUAL", withholding_tax_rate=rate)


class TestAgainstGeneratedData:
    """At scale, on data that moves."""

    @pytest.fixture(scope="class")
    def levels(self):
        from beacon.synthetic import SyntheticConfig, generate

        dataset = generate(SyntheticConfig(assets=12, start="2022-01-03",
                                           end="2024-06-28", seed=5))
        fetcher = dataset.fetcher()
        names = list(dataset.universe.index)

        def calculate(return_type, rate=0.0):
            definition = IndexDefinition(
                index_id="I", index_name="I", base_date="2022-01-03",
                base_value=1000.0, currency="USD", eligibility_rules=[],
                weighting_scheme=EqualWeighted(),
                rebalancing_frequency="QUARTERLY", universe_identifiers=names,
                return_type=return_type, withholding_tax_rate=rate)

            return IndexCalculator(definition, fetcher).run(
                start_date="2022-01-03", end_date="2024-06-28").index_levels

        return {"price": calculate(PRICE),
                "gross": calculate(TOTAL_RETURN),
                "net": calculate(NET_TOTAL_RETURN, 0.30)}

    def test_the_three_are_ordered_throughout(self, levels):
        assert (levels["gross"] >= levels["net"] - 1e-9).all()
        assert (levels["net"] >= levels["price"] - 1e-9).all()

    def test_the_yield_pickup_is_plausible(self, levels):
        """A universe where most names yield a few percent should give a
        total-return premium of roughly one to two percent a year. A wildly
        larger figure would mean distributions were being counted more than
        once — which is the failure mode a direction test cannot see."""
        def annualised(series):
            return (series.iloc[-1] / series.iloc[0]) ** (252 / len(series)) - 1

        pickup = annualised(levels["gross"]) - annualised(levels["price"])

        assert 0.002 < pickup < 0.04, f"{pickup:.4f} per year"

    def test_they_start_together(self, levels):
        """Whatever the type, the base value is the base value."""
        assert levels["price"].iloc[0] == levels["gross"].iloc[0]
