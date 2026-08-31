# tests/test_survivorship.py
"""BN-130: listings, delistings, and the point-in-time universe.

Two separate claims are tested here and they need different evidence.

**The mechanism** — that deleting a constituent leaves the index level
continuous — is exact, so it is tested on a hand-built two-name index where
the arithmetic can be checked by hand. A generated panel would prove it only
approximately and would blame a real defect on noise.

**The bias** — that an index built only from survivors outperforms one built
as it went along — is *measured* rather than tested, and the measurement lives
in the worklog: **+0.42%/yr** over 300 names and ten years, 8.45% for a
survivors-only index against 8.04% point-in-time.

It is not a test here because it cannot honestly be one at a size the suite
can afford. Seeing the effect takes two full index calculations over a few
hundred names, and the calculator costs roughly one market-data slice per
constituent per day — comfortably past the 120-second timeout this project
sets deliberately. Shrunk to fit, the effect sits inside the noise and the
assertion passes or fails on the seed, which is worse than not asserting it.

The effect is small for two reasons worth knowing, and both are consequences
of decisions that are right in isolation. Cap weighting mutes it: a company on
its way to failing shrinks first, so by the time it goes it carries almost no
weight. And the divisor deletion means the index never books the final
collapse, so a point-in-time index gets much of the same protection a
survivors-only one does.

What would make it large enough to assert cheaply is delisting driven by a
name's *realised* path rather than its drawn alpha — a company delists because
it has already fallen, and that fall belongs in the index. That needs the
lives drawn after the returns are simulated, which reorders `generate`.
"""

import numpy as np
import pandas as pd
import pytest

from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted, MarketCapWeighted
from beacon.synthetic import SyntheticConfig, generate
from beacon.synthetic import listings as listings_module
from beacon.synthetic import universe as universe_module

START = "2021-01-04"
END = "2021-03-31"

# The generated panel every statistical test below shares. One panel rather
# than one per class: generating 120 names over ten years is the most
# expensive thing in this file, and three copies of it bought nothing.
# Sized by measurement, not by instinct. The bias test runs *two* full index
# calculations over this panel, and the calculator costs roughly one
# market-data slice per constituent per day -- so its runtime is the product
# of the two numbers below, not a function of either. At 250 names over ten
# years this file alone ran longer than the rest of the suite put together.
#
# Sixty names over five years still retires enough of the universe for the
# direction of the bias to clear the noise, which is all this asserts.
PANEL_ASSETS = 60
PANEL_START = "2017-01-03"
PANEL_END = "2021-12-31"
PANEL_SEED = 3


@pytest.fixture(scope="module")
def panel():
    """A universe with listings and delistings in it."""
    return generate(SyntheticConfig(assets=PANEL_ASSETS, start=PANEL_START,
                                    end=PANEL_END, seed=PANEL_SEED))


def build_fetcher(delist_on: str | None) -> DataFetcher:
    """Two names on a flat, known price path; one may stop being listed.

    Prices are constant so any movement in the index level is the mechanism
    under test rather than the market.
    """
    dates = pd.bdate_range(START, END)

    rows = []
    for identifier, price in (("AAA", 100.0), ("BBB", 50.0)):
        for date in dates:
            if identifier == "BBB" and delist_on and date > pd.Timestamp(delist_on):
                continue

            rows.append({"IDENTIFIER": identifier, "DATE": date,
                         "OPEN": price, "HIGH": price, "LOW": price,
                         "CLOSE": price, "VOLUME": 1_000.0,
                         "SHARES_OUTSTANDING": 1_000_000.0,
                         "FREE_FLOAT": 1.0})

    reference = pd.DataFrame([
        {"IDENTIFIER": "AAA", "DATE_FROM": START, "DATE_TO": pd.NaT,
         "NAME": "A", "CURRENCY": "USD", "EXCHANGE": "XNAS"},
        {"IDENTIFIER": "BBB", "DATE_FROM": START,
         "DATE_TO": pd.Timestamp(delist_on) if delist_on else pd.NaT,
         "NAME": "B", "CURRENCY": "USD", "EXCHANGE": "XNAS"},
    ])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def level_series(fetcher: DataFetcher) -> pd.Series:
    definition = IndexDefinition(
        index_id="T", index_name="T", base_date=START, base_value=1000.0,
        currency="USD", eligibility_rules=[],
        weighting_scheme=EqualWeighted(),
        rebalancing_frequency="QUARTERLY",
        universe_identifiers=["AAA", "BBB"])

    return IndexCalculator(definition, fetcher).run(start_date=START,
                                                    end_date=END).index_levels


class TestTheMechanism:
    """A deletion is a divisor adjustment, and the level does not step."""

    def test_a_delisting_leaves_the_level_unchanged(self):
        """The claim in one assertion.

        Prices never move, so a correct index is flat at its base value for
        the whole quarter whether or not half of it is delisted half way
        through. Without the divisor adjustment the level drops to the
        surviving name's share on the day BBB goes -- it reports a loss no
        holder took, because in reality the position was sold at a price and
        the proceeds stayed in the fund.
        """
        levels = level_series(build_fetcher(delist_on="2021-02-15"))

        assert levels.notna().all()
        assert np.allclose(levels.to_numpy(), 1000.0), (
            f"level moved on a flat market: "
            f"min {levels.min():.2f}, max {levels.max():.2f}")

    def test_it_matches_the_index_that_never_lost_a_name(self):
        """Same prices, same weights, one with a deletion and one without."""
        deleted = level_series(build_fetcher(delist_on="2021-02-15"))
        intact = level_series(build_fetcher(delist_on=None))

        pd.testing.assert_series_equal(deleted, intact)

    def test_the_divisor_absorbs_it(self):
        """The level is continuous *because* the divisor moved, not because
        nothing happened. A test that only checked the level would pass on an
        index that quietly ignored the delisting and kept pricing a dead
        holding at its last close forever."""
        definition = IndexDefinition(
            index_id="T", index_name="T", base_date=START, base_value=1000.0,
            currency="USD", eligibility_rules=[],
            weighting_scheme=EqualWeighted(),
            rebalancing_frequency="QUARTERLY",
            universe_identifiers=["AAA", "BBB"])

        result = IndexCalculator(
            definition, build_fetcher(delist_on="2021-02-15")).run(
                start_date=START, end_date=END)

        divisors = result.divisor_history

        assert divisors.iloc[-1] < divisors.iloc[0], (
            "the divisor never moved, so the deletion was not applied")

    def test_an_index_with_no_delistings_keeps_a_constant_divisor(self):
        """The cost of the check, when there is nothing to check."""
        result_divisors = IndexCalculator(
            IndexDefinition(
                index_id="T", index_name="T", base_date=START,
                base_value=1000.0, currency="USD", eligibility_rules=[],
                weighting_scheme=EqualWeighted(),
                rebalancing_frequency="QUARTERLY",
                universe_identifiers=["AAA", "BBB"]),
            build_fetcher(delist_on=None)).run(
                start_date=START, end_date=END).divisor_history

        assert result_divisors.nunique() == 1


@pytest.mark.timeout(900)
class TestTheGeneratedLives:
    """What the generator now emits. Shares the panel, so it pays for it."""

    def test_some_names_leave_and_some_arrive(self, panel):
        universe = panel.universe

        assert universe["listed_to"].notna().any(), "nothing ever delists"
        assert (universe["listed_from"] > universe["listed_from"].min()).any(), (
            "nothing ever lists partway through")

    def test_the_reference_data_carries_the_life(self, panel):
        reference = panel.reference.data

        assert reference["DATE_TO"].notna().any()
        assert reference["DATE_FROM"].nunique() > 1

    def test_market_data_stops_at_the_delisting(self, panel):
        """A delisted name has no rows afterwards, rather than rows of NaN.

        The two are different claims: a null price says the market was open
        and the quote is missing, which somebody should chase; no row says the
        company was not listed, which is a fact.
        """
        universe = panel.universe
        market = panel.market.data

        for identifier in universe.index[universe["listed_to"].notna()][:5]:
            last = market.loc[identifier].index.max()

            assert last <= universe.loc[identifier, "listed_to"]

    def test_market_data_starts_at_the_listing(self, panel):
        universe = panel.universe
        market = panel.market.data
        late = universe.index[universe["listed_from"] > universe["listed_from"].min()]

        for identifier in late[:5]:
            first = market.loc[identifier].index.min()

            assert first >= universe.loc[identifier, "listed_from"]

    def test_point_in_time_resolution_drops_a_delisted_name(self, panel):
        """The reason `DATE_TO` is worth emitting at all."""
        universe = panel.universe
        gone = universe.index[universe["listed_to"].notna()][0]
        left_on = universe.loc[gone, "listed_to"]

        fetcher = panel.fetcher()

        before = fetcher.fetch_reference_data(
            gone, (left_on - pd.Timedelta(days=7)).strftime("%Y-%m-%d"))
        after = fetcher.fetch_reference_data(
            gone, (left_on + pd.Timedelta(days=30)).strftime("%Y-%m-%d"))

        assert not before.empty
        assert after.empty

    def test_delistings_cluster_in_the_crises(self):
        """A constant hazard would spread them evenly, and survivorship bias
        would be a slow drip rather than the thing that eats a backtest.

        Real companies fail together, inside the drawdown that is already
        happening -- which is exactly when a survivors-only universe is most
        misleading.
        """
        dates = pd.bdate_range("2000-01-03", "2024-12-31")
        drawn = listings_module.draw(600, dates, np.random.default_rng(3))

        by_year = drawn["listed_to"].dropna().dt.year.value_counts()
        crisis_years = [2001, 2002, 2008, 2009, 2020, 2022]

        crisis_rate = by_year.reindex(crisis_years).fillna(0).mean()
        calm_rate = by_year.drop(crisis_years, errors="ignore").mean()

        assert crisis_rate > calm_rate * 1.4, (
            f"crisis years average {crisis_rate:.1f} delistings against "
            f"{calm_rate:.1f} in calm years")

    def test_the_rates_can_be_turned_off(self):
        """Every dataset generated before BN-130 had a constant universe, and
        a caller that wants one back must be able to say so."""
        dates = pd.bdate_range("2015-01-02", "2024-12-31")
        drawn = listings_module.draw(200, dates, np.random.default_rng(1),
                                     delisting_rate=0.0, listing_rate=0.0)

        assert drawn["listed_to"].isna().all()
        assert (drawn["listed_from"] == dates[0]).all()

    def test_a_name_never_leaves_before_it_arrives(self):
        dates = pd.bdate_range("2000-01-03", "2024-12-31")
        drawn = listings_module.draw(800, dates, np.random.default_rng(5))
        both = drawn.dropna()

        assert (both["listed_to"] > both["listed_from"]).all()

    def test_a_universe_built_without_dates_is_fully_listed(self):
        """The signature stays usable for callers that only want the static
        fields, and they get the old behaviour rather than an error."""
        drawn = universe_module.build(20, np.random.default_rng(2))

        assert "listed_to" not in drawn


@pytest.mark.timeout(900)
class TestTheEngineConvertsCurrency:
    """Found while building survivorship, and not a survivorship bug.

    `IndexCalculator` has always converted market values into the index
    currency. `BacktestEngine` did not: it returned the raw stored close, so a
    300 yen share was valued as 300 dollars. Against the single-currency
    universe that existed until BN-128 every rate was 1.0 and the omission was
    invisible; the moment the generator grew regions it showed up as 2.38%
    annualised tracking error against an index the portfolio should have
    tracked almost exactly.
    """

    @pytest.fixture(scope="module")
    def dataset(self):
        """Deliberately short and small: this class is about a conversion,
        not about a long price path."""
        return generate(SyntheticConfig(assets=40, start="2021-01-04",
                                        end="2021-12-31", seed=3))

    def test_a_foreign_price_is_converted(self, dataset):
        from beacon.backtest.engine import BacktestEngine

        engine = BacktestEngine(
            start_date="2021-01-04", end_date="2021-12-31",
            initial_capital=1e7, data_provider=dataset.fetcher(),
            target_weights={pd.Timestamp("2021-01-04"): {}})

        universe = dataset.universe
        foreign = universe.index[universe["CURRENCY"] == "JPY"]

        if not len(foreign):
            pytest.skip("no JPY name in this universe")

        date = pd.Timestamp("2021-06-15")
        identifier = foreign[0]

        raw = float(dataset.fetcher().fetch_market_data(
            identifier, "2021-06-15", "2021-06-15")["CLOSE"].iloc[0])
        converted = engine._fetch_price(identifier, date)

        assert converted is not None
        assert converted < raw / 50, (
            f"a yen price of {raw:.2f} came back as {converted:.2f}; "
            f"it was not converted")

    def test_a_domestic_price_is_untouched(self, dataset):
        from beacon.backtest.engine import BacktestEngine

        engine = BacktestEngine(
            start_date="2021-01-04", end_date="2021-12-31",
            initial_capital=1e7, data_provider=dataset.fetcher(),
            target_weights={pd.Timestamp("2021-01-04"): {}})

        universe = dataset.universe
        domestic = universe.index[universe["CURRENCY"] == "USD"][0]

        raw = float(dataset.fetcher().fetch_market_data(
            domestic, "2021-06-15", "2021-06-15")["CLOSE"].iloc[0])
        converted = engine._fetch_price(domestic, pd.Timestamp("2021-06-15"))

        assert converted == pytest.approx(raw, rel=1e-6)

    def test_the_rate_moves_with_the_date(self, dataset):
        """A *fixed* rate would look like a conversion and change nothing.

        The engine sizes a position by value, so `quantity x price x rate` is
        the target value whatever the rate is -- a constant scale factor
        cancels out of the weights entirely. Only a rate that moves gives a
        foreign holding the currency exposure it actually has, which is why
        this asserts the two dates differ rather than merely that a rate was
        applied.
        """
        from beacon.backtest.engine import BacktestEngine

        engine = BacktestEngine(
            start_date="2021-01-04", end_date="2021-12-31",
            initial_capital=1e7, data_provider=dataset.fetcher(),
            target_weights={pd.Timestamp("2021-01-04"): {}})

        universe = dataset.universe
        foreign = universe.index[universe["CURRENCY"] != "USD"]

        if not len(foreign):
            pytest.skip("no foreign name in this universe")

        early = engine._rate_for(foreign[0], pd.Timestamp("2021-02-01"))
        late = engine._rate_for(foreign[0], pd.Timestamp("2021-11-01"))

        assert early != late, "the rate is constant, so it cancels"

    def test_it_tracks_the_index_it_is_given(self, dataset):
        """The symptom, end to end. A portfolio handed an index's own weights
        and charged no costs should track it to a few basis points."""
        from beacon.backtest.engine import BacktestEngine

        fetcher = dataset.fetcher()
        definition = IndexDefinition(
            index_id="FX", index_name="FX", base_date="2021-01-04",
            base_value=1000.0, currency="USD", eligibility_rules=[],
            weighting_scheme=MarketCapWeighted(use_free_float=True),
            rebalancing_frequency="QUARTERLY",
            universe_identifiers=list(dataset.universe.index),
            max_constituent_weight=0.10)

        index = IndexCalculator(definition, fetcher).run(
            start_date="2021-01-04", end_date="2021-12-31")

        backtest = BacktestEngine(
            start_date="2021-01-04", end_date="2021-12-31",
            initial_capital=1e7, data_provider=fetcher,
            index_result=index, transaction_cost_bps=0.0).run()

        tracking = backtest.summary()["tracking_error"]

        assert tracking is not None
        assert tracking < 0.01, (
            f"tracking error {tracking:.2%} against an index the portfolio "
            f"was handed the weights of")
