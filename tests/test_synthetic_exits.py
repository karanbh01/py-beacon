# tests/test_synthetic_exits.py
"""BN-360: generated names leave as a large-cap market's do. Smaller names
leave more often, larger ones mostly through takeovers announced in advance
and paid at the offer, and failures, not bids, crowd into a crisis."""
import numpy as np
import pandas as pd
import pytest

from beacon.synthetic import SyntheticConfig, exits
from beacon.synthetic.dataset import generate

CONFIG = SyntheticConfig(assets=400, seed=5, start="2016-01-04", end="2025-12-31")


@pytest.fixture(scope="module")
def dataset():
    return generate(CONFIG)


def size_fifths(universe: pd.DataFrame) -> pd.Series:
    return pd.qcut(universe["market_cap"].rank(method="first"), 5,
                   labels=["smallest", "2", "3", "4", "largest"])


class TestWhoLeaves:

    def test_smaller_names_leave_more_often(self, dataset):
        universe = dataset.universe
        rates = universe["listed_to"].notna().groupby(size_fifths(universe),
                                                      observed=True).mean()

        assert rates["smallest"] > 2 * rates["largest"]

    def test_larger_names_mostly_leave_through_takeovers(self, dataset):
        universe = dataset.universe
        gone = universe[universe["listed_to"].notna()]
        takeovers = gone["exit"].eq(exits.TAKEOVER).groupby(size_fifths(universe)[gone.index],
                                                           observed=True).mean()

        assert takeovers["largest"] > 0.6
        assert takeovers["smallest"] < takeovers["largest"]


class TestATakeover:

    def test_its_price_jumps_on_the_bid_and_closes_at_the_offer(self, dataset):
        universe = dataset.universe
        market = dataset.market.data

        for name, deal in universe[universe["exit"].eq(exits.TAKEOVER)].head(5).iterrows():
            closes = market.loc[name, "CLOSE"]
            before = closes[closes.index < deal["announced"]].iloc[-1]
            last = closes.loc[:deal["listed_to"]].iloc[-1]
            on_bid = closes.loc[deal["announced"]]

            assert on_bid / before - 1 == pytest.approx(
                (1 + deal["premium"]) * (1 - deal["spread"]) - 1, abs=0.06)
            assert last / before - 1 == pytest.approx(deal["premium"], abs=0.06)

    def test_it_is_announced_ahead_of_its_last_day(self, dataset):
        deals = dataset.universe[dataset.universe["exit"].eq(exits.TAKEOVER)]
        gap = (deals["listed_to"] - deals["announced"]).dt.days

        assert len(deals) > 0
        assert (gap >= exits.ANNOUNCEMENT_SESSIONS[0]).all()


class TestTheModel:

    def test_smallness_has_a_mean_of_one_and_falls_with_size(self):
        weight = exits.smallness(np.array([1e9, 1e10, 1e11]))

        assert weight.mean() == pytest.approx(1.0)
        assert weight[0] > weight[1] > weight[2]
