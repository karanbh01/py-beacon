# tests/test_strategy_synthetic.py
"""BN-270: synthetic replication, a substitute basket and a total return swap.

Ten index names and six basket names, from two unrelated factors, so the
basket's path is nothing like the index's. The swap should hand the fund the
index's return whatever the basket does, less its spread, and reset at each
rebalance, on flow days and at its exposure limit.
"""
import numpy as np
import pandas as pd
import pytest

from beacon import catalogue
from beacon.backtest import Backtest, DatedFlows
from beacon.backtest.swap import FLOWS, LIMIT, REBALANCE
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted, MarketCapWeighted
from beacon.index.schedule import sessions
from beacon.strategy import IndexTracking, SyntheticReplication
from beacon.strategy.synthetic import UCITS_EXPOSURE_LIMIT

DAYS = sessions(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-12-31"), "XNYS")
INDEX_NAMES = [f"IDX{i}" for i in range(10)]
BASKET_NAMES = [f"BSK{i}" for i in range(6)]
START, END = "2024-01-02", "2024-12-31"
FLOW_DAY = DAYS[100]


def fetcher() -> DataFetcher:
    random = np.random.default_rng(7)
    index_factor = random.normal(0.0006, 0.011, len(DAYS))
    basket_factor = random.normal(-0.0002, 0.013, len(DAYS))
    rows = []

    for position, name in enumerate([*INDEX_NAMES, *BASKET_NAMES]):
        factor = index_factor if name.startswith("IDX") else basket_factor
        prices = 40.0 * np.cumprod(1.0 + factor + random.normal(0.0, 0.006, len(DAYS)))

        for day, price in zip(DAYS, prices, strict=True):
            rows.append({"IDENTIFIER": name, "DATE": day, "CLOSE": price,
                         "SHARES_OUTSTANDING": 1e9 / (1 + position)})

    reference = pd.DataFrame([
        {"IDENTIFIER": name, "NAME": name, "DATE_FROM": "2020-01-01",
         "CURRENCY": "USD", "EXCHANGE": "XNYS"}
        for name in [*INDEX_NAMES, *BASKET_NAMES]])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


DATA = fetcher()


def index(index_id: str = "MCW",
          names: list[str] | None = None) -> IndexDefinition:
    return IndexDefinition(index_id=index_id, index_name=index_id, base_date=START,
                           base_value=1000.0, currency="USD", eligibility_rules=[],
                           weighting_scheme=MarketCapWeighted(),
                           rebalancing_frequency="QUARTERLY", calendar="XNYS",
                           universe_identifiers=names or INDEX_NAMES)


BASKET = dict.fromkeys(BASKET_NAMES, 1.0)


def run(replication, **settings):
    return Backtest(initial_capital=1e7, data_provider=DATA, cache=False,
                    **settings).run(IndexTracking(index(), replication), start=START, end=END)


def growth(series: pd.Series) -> float:
    return float(series.iloc[-1] / series.iloc[0])


@pytest.fixture(scope="module")
def synthetic():
    return run(SyntheticReplication(BASKET))


class TestConstruction:

    def test_the_basket_is_required(self):
        with pytest.raises(TypeError):
            SyntheticReplication()  # type: ignore[call-arg]

    def test_an_empty_basket_is_refused(self):
        with pytest.raises(ValueError, match="at least one name"):
            SyntheticReplication({})

    def test_a_weight_that_is_not_positive_is_refused(self):
        with pytest.raises(ValueError, match="BSK1"):
            SyntheticReplication({"BSK0": 1.0, "BSK1": 0.0})

    @pytest.mark.parametrize("limit", [0.0, -0.1, 1.5])
    def test_the_limit_is_a_share_of_net_assets(self, limit):
        with pytest.raises(ValueError, match="exposure_limit"):
            SyntheticReplication(BASKET, exposure_limit=limit)

    def test_the_spread_must_be_a_number(self):
        with pytest.raises(ValueError, match="spread_bps"):
            SyntheticReplication(BASKET, spread_bps=float("nan"))

    def test_fixed_weights_are_scaled_to_one(self):
        replication = SyntheticReplication({"BSK0": 3.0, "BSK1": 1.0})
        step = replication.replicate({}, pd.Timestamp(START), None)  # type: ignore[arg-type]

        assert step.weights == {"BSK0": 0.75, "BSK1": 0.25}

    def test_the_ucits_limit_is_the_default(self):
        assert SyntheticReplication(BASKET).terms.exposure_limit == UCITS_EXPOSURE_LIMIT


class TestTheFundEarnsTheIndex:

    def test_the_fund_holds_the_basket_not_the_index(self, synthetic):
        held = set(synthetic.portfolio.holdings)

        assert held == set(BASKET_NAMES)
        assert synthetic.replication[0].weights == pytest.approx(
            {name: 1 / len(BASKET_NAMES) for name in BASKET_NAMES})

    def test_it_tracks_the_index_although_the_basket_went_elsewhere(self, synthetic):
        levels = synthetic.index.target.levels
        physical_basket = Backtest(initial_capital=1e7, data_provider=DATA, cache=False).run(
            index("BASKET", BASKET_NAMES), start=START, end=END)

        fund = growth(synthetic.nav_per_unit)
        tracked = growth(levels)

        # The basket on its own wanders far from the index...
        basket_path = physical_basket.trading_nav / physical_basket.trading_nav.iloc[0]
        index_path = (levels / levels.iloc[0]).reindex(basket_path.index)
        assert (basket_path - index_path).abs().max() > 0.10
        # ...but through the swap the fund ends where the index does.
        assert fund == pytest.approx(tracked, abs=1e-6)

    def test_the_spread_is_what_the_fund_gives_up(self, synthetic):
        spread = run(SyntheticReplication(BASKET, spread_bps=50.0))
        resets = spread.swap_resets
        ends = [reset.date for reset in resets[1:]] + [DAYS[-1]]

        levels = spread.index.target.levels

        # 50bp a year over calendar days (ACT/360), on each period's notional,
        # which follows the fund. What a period's spread costs at the end of
        # the run is what that money would have become in the index since.
        expected = sum(reset.notional * 0.005 * (end - reset.date).days / 360
                       * levels.iloc[-1] / levels.asof(end)
                       for reset, end in zip(resets, ends, strict=True)) / 1e7
        drag = growth(synthetic.nav_per_unit) - growth(spread.nav_per_unit)

        assert drag == pytest.approx(expected, rel=0.02)

    def test_the_swap_is_part_of_the_nav(self, synthetic):
        nav = synthetic.trading_nav
        swap = synthetic.swap_value.reindex(nav.index)
        holdings_and_cash = synthetic.portfolio.nav.reindex(nav.index)

        assert (nav - holdings_and_cash).to_numpy() == pytest.approx(swap.to_numpy(), abs=1e-6)
        assert swap.abs().max() > 0.0

    def test_a_run_without_a_swap_has_none(self):
        plain = Backtest(initial_capital=1e7, data_provider=DATA, cache=False).run(
            index(), start=START, end=END)

        assert plain.swap_value.empty
        assert plain.swap_resets == []
        assert plain.counterparty_exposure.empty


class TestResets:

    def test_it_resets_at_every_rebalance(self, synthetic):
        rebalances = [step.date for step in synthetic.replication]
        resets = [reset.date for reset in synthetic.swap_resets
                  if reset.reason == REBALANCE]

        assert resets == rebalances

    def test_a_reset_settles_the_swap_in_cash(self, synthetic):
        later = synthetic.swap_resets[1:]
        settled = [flow for flow in synthetic.portfolio.cash_flows if flow.kind == "SWAP"]

        assert later
        assert [flow.amount for flow in settled] == pytest.approx(
            [reset.settled for reset in later if reset.settled != 0.0])
        assert all(synthetic.swap_value[reset.date] == pytest.approx(0.0, abs=1e-6)
                   for reset in later)

    def test_it_resets_when_the_exposure_reaches_the_limit(self):
        limited = run(SyntheticReplication(BASKET, exposure_limit=0.02))
        exposure = limited.counterparty_exposure

        assert any(reset.reason == LIMIT for reset in limited.swap_resets)
        # Reset the day the limit is reached, so no close is left above it.
        assert exposure.max() < 0.02

    def test_the_exposure_is_what_the_fund_is_owed(self, synthetic):
        exposure = synthetic.counterparty_exposure

        assert (exposure >= 0.0).all()
        assert exposure.max() > 0.0

    def test_it_resets_when_money_flows(self):
        flowed = run(SyntheticReplication(BASKET),
                     flows=DatedFlows({FLOW_DAY: 2e6}))
        reset = next(reset for reset in flowed.swap_resets if reset.date == FLOW_DAY)

        assert reset.reason == FLOWS
        # The new notional includes the money that arrived.
        assert reset.notional == pytest.approx(flowed.trading_nav[FLOW_DAY], rel=1e-9)

    def test_the_fund_still_earns_the_index_with_flows(self):
        flowed = run(SyntheticReplication(BASKET), flows=DatedFlows({FLOW_DAY: 2e6}))

        assert growth(flowed.nav_per_unit) == pytest.approx(
            growth(flowed.index.target.levels), abs=1e-6)


class TestABasketIndex:

    def test_the_basket_can_be_an_index(self):
        basket = IndexDefinition(index_id="BASKET", index_name="Basket", base_date=START,
                                 base_value=100.0, currency="USD", eligibility_rules=[],
                                 weighting_scheme=EqualWeighted(),
                                 rebalancing_frequency="QUARTERLY", calendar="XNYS",
                                 universe_identifiers=BASKET_NAMES)
        result = run(SyntheticReplication(basket))

        assert set(result.portfolio.holdings) == set(BASKET_NAMES)
        assert growth(result.nav_per_unit) == pytest.approx(
            growth(result.index.target.levels), abs=1e-6)

    def test_an_uncalculated_basket_index_says_so(self):
        replication = SyntheticReplication(index("BASKET", BASKET_NAMES))

        with pytest.raises(ValueError, match="not been calculated"):
            replication.replicate({}, pd.Timestamp(START), None)  # type: ignore[arg-type]


class TestTheCatalogue:

    def test_the_editor_can_offer_it(self):
        entry = catalogue.entry_for(catalogue.REPLICATION, "SyntheticReplication")
        parameters = {parameter.name: parameter for parameter in entry.parameters}

        assert parameters["basket"].required
        assert parameters["exposure_limit"].default == UCITS_EXPOSURE_LIMIT

    def test_its_settings_round_trip(self):
        from beacon.server.backtest_settings import typed_payload

        replication = SyntheticReplication({"BSK0": 1.0}, spread_bps=20.0,
                                           exposure_limit=0.05)
        payload = typed_payload(catalogue.REPLICATION, replication)

        assert payload == {"type": "SyntheticReplication",
                           "params": {"basket": {"BSK0": 1.0}, "spread_bps": 20.0,
                                      "exposure_limit": 0.05}}
