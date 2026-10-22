# tests/test_server_derivative_records.py
"""Contract tests for a fund's swap and futures records on the API (BN-315).

Three names through 2024 and an equal-weighted index over them, held three
ways: through a swap on a basket of two of them, through rolled index futures,
and physically with a cash buffer equitised with futures.
"""
import copy

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.index.schedule import sessions
from beacon.server import ServerConfig, create_app
from beacon.server.jobs import SUCCEEDED

TOKEN = "test-token-value"
DAYS = sessions(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-12-31"), "XNYS")
NAMES = ["A", "B", "C"]
START, END = "2024-01-02", "2024-12-31"

INDEX = {
    "id": "EW3",
    "name": "Equal three",
    "base_date": START,
    "base_value": 100.0,
    "currency": "USD",
    "calendar": "XNYS",
    "rebalancing_frequency": "QUARTERLY",
    "universe": {"universe_id": None, "identifiers": NAMES},
    "pipeline": {
        "selection": [],
        "weighting": {"id": "weighting", "scheme": "EqualWeighted",
                      "params": {}, "max_weight": None},
        "treatment": {"corporate_actions": "ADJUST_DIVISOR"},
    },
}

STRATEGIES = {
    "synthetic": {"type": "index_tracking",
                  "replication": {"type": "SyntheticReplication",
                                  "params": {"basket": {"A": 0.5, "B": 0.5},
                                             "spread_bps": 10}}},
    "futures": {"type": "index_tracking",
                "replication": {"type": "FuturesReplication",
                                "params": {"roll_days": 5, "roll_cost_bps": 2}}},
}


def build_fetcher() -> DataFetcher:
    """Three names on random paths with a common market move."""
    random = np.random.default_rng(37)
    market = random.normal(0.0004, 0.009, len(DAYS))
    rows = []

    for name in NAMES:
        prices = 50.0 * np.cumprod(1.0 + market + random.normal(0.0, 0.006, len(DAYS)))
        rows += [{"IDENTIFIER": name, "DATE": day, "CLOSE": price, "SHARES_OUTSTANDING": 1e8,
                  "VOLUME": 1e6}
                 for day, price in zip(DAYS, prices, strict=True)]

    reference = pd.DataFrame([{"IDENTIFIER": name, "NAME": name, "DATE_FROM": "2020-01-01",
                               "CURRENCY": "USD", "EXCHANGE": "XNYS"} for name in NAMES])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A client holding the index and a strategy for each way of holding it."""
    config = ServerConfig(auth_token=TOKEN, data_fetcher=build_fetcher(),
                          storage_root=tmp_path_factory.mktemp("records"))
    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        assert entered.post("/indices", json=copy.deepcopy(INDEX),
                            headers=auth()).status_code == 200

        for strategy_id, strategy in STRATEGIES.items():
            saved = entered.post("/strategies", json={"id": strategy_id, "name": strategy_id,
                                                      "index_id": "EW3",
                                                      "strategy": strategy},
                                 headers=auth())
            assert saved.status_code == 200, saved.text

        equitised = entered.post("/strategies", json={
            "id": "equitised", "name": "equitised", "index_id": "EW3",
            "implementation": {"cash_buffer": 0.05, "equitise": {"roll_days": 5}}},
            headers=auth())
        assert equitised.status_code == 200, equitised.text
        yield entered


def run(client,
        strategy_id: str) -> dict:
    """Backtest a strategy over the year and return its result."""
    response = client.post(f"/strategies/{strategy_id}/backtest",
                           json={"start": START, "end": END,
                                 "modelling_assumptions": {"cash_rate": 0.04}},
                           headers=auth())
    assert response.status_code == 202, response.text
    client.portal.call(client.app.state.jobs.drain)
    job = client.get(f"/jobs/{response.json()['job_id']}", headers=auth()).json()
    assert job["status"] == SUCCEEDED, job

    return job["result"]


@pytest.fixture(scope="module")
def results(client) -> dict[str, dict]:
    """Each strategy's run."""
    return {strategy_id: run(client, strategy_id)
            for strategy_id in ("synthetic", "futures", "equitised")}


class TestTheSwap:

    def test_its_value_and_exposure_share_the_level_s_dates(self,
                                                            results):
        synthetic = results["synthetic"]

        assert synthetic["swap_value"]["index"] == synthetic["level"]["index"]
        assert synthetic["counterparty_exposure"]["index"] == synthetic["level"]["index"]

    def test_the_exposure_is_what_the_fund_is_owed(self,
                                                   results):
        """Never negative, and never above the 10% limit by more than a day's move."""
        exposure = results["synthetic"]["counterparty_exposure"]["data"]

        assert min(exposure) >= 0.0
        assert max(exposure) < 0.15

    def test_each_reset_says_when_why_and_what_settled(self,
                                                       results):
        resets = results["synthetic"]["swap_resets"]

        assert resets[0]["settled"] == 0.0
        assert {"rebalance"} <= {reset["reason"] for reset in resets}
        assert all(reset["notional"] > 0 for reset in resets)


class TestTheFutures:

    @pytest.mark.parametrize("strategy_id", ["futures", "equitised"])
    def test_their_notional_and_exposure_share_the_level_s_dates(self,
                                                                 results,
                                                                 strategy_id):
        result = results[strategy_id]

        assert result["futures_notional"]["index"] == result["level"]["index"]
        assert result["futures_exposure"]["index"] == result["level"]["index"]

    def test_a_futures_fund_is_exposed_to_about_its_net_assets(self,
                                                              results):
        exposure = results["futures"]["futures_exposure"]["data"]

        assert exposure[-1] == pytest.approx(1.0, abs=0.05)

    def test_an_equitised_buffer_is_exposed_to_about_the_buffer(self,
                                                                results):
        exposure = results["equitised"]["futures_exposure"]["data"]

        assert exposure[-1] == pytest.approx(0.05, abs=0.02)

    def test_each_roll_carries_its_contracts_price_and_cost(self,
                                                            results):
        rolls = results["futures"]["futures_rolls"]

        # March, June, September and December: the last rolls on 13 December,
        # five sessions before its expiry, into March 2025.
        assert [roll["date"][:7] for roll in rolls] == ["2024-03", "2024-06", "2024-09",
                                                        "2024-12"]
        for roll in rolls:
            assert roll["contracts"] == int(roll["contracts"])
            assert (roll["contracts"] * roll["price"] * roll["multiplier"]
                    == pytest.approx(roll["notional"]))
            assert roll["cost"] > 0
            assert roll["expiry"] > roll["date"]


class TestWithoutThem:

    def test_a_physical_fund_answers_as_before(self,
                                               client):
        plain = client.post("/strategies", json={"id": "plain", "name": "plain",
                                                 "index_id": "EW3"}, headers=auth())
        assert plain.status_code == 200
        result = run(client, "plain")

        for name in ("swap_value", "counterparty_exposure", "futures_notional",
                     "futures_exposure"):
            assert result[name] is None, name
        assert result["swap_resets"] == [] and result["futures_rolls"] == []

    def test_a_synthetic_fund_has_no_futures(self,
                                             results):
        assert results["synthetic"]["futures_notional"] is None
        assert results["futures"]["swap_value"] is None
