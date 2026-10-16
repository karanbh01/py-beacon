# tests/test_server_index_level.py
"""Contract tests for `GET /indices/{index_id}/level` (BN-316): an index on
its own, with no backtest."""
import copy

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.backtest.main import Backtest
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.server import ServerConfig, create_app
from beacon.server.definitions import build_definition
from beacon.server.schemas import IndexDocument

TOKEN = "test-token-value"

START = "2023-10-02"
END = "2024-03-29"
DATES = pd.bdate_range(START, END)
# The index runs on its calendar: 29 March 2024 was Good Friday, when the
# exchange was shut.
LAST_SESSION = "2024-03-28"

# Two names on deterministic geometric paths, so the level is known.
GROWTH = {"AAA": 1.30, "BBB": 0.85}
BASE_PRICE = {"AAA": 100.0, "BBB": 50.0}
SHARES = 1_000


def build_fetcher() -> DataFetcher:
    """Synthetic market and reference data with no randomness."""
    span = len(DATES) - 1
    rows = [{"IDENTIFIER": name,
             "DATE": date,
             "CLOSE": BASE_PRICE[name] * (growth ** (position / span)),
             "VOLUME": 1_000_000,
             "SHARES_OUTSTANDING": SHARES}
            for name, growth in GROWTH.items()
            for position, date in enumerate(DATES)]
    reference = pd.DataFrame([{"IDENTIFIER": name, "DATE_FROM": "2020-01-01",
                               "NAME": name, "CURRENCY": "USD", "EXCHANGE": "NYSE"}
                              for name in GROWTH])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


INDEX = {
    "id": "EQ2",
    "name": "Equal two",
    "base_date": START,
    "base_value": 1000.0,
    "currency": "USD",
    "calendar": "XNYS",
    "rebalancing_frequency": "QUARTERLY",
    "universe": {"universe_id": None, "identifiers": list(GROWTH)},
    "pipeline": {
        "selection": [],
        "weighting": {"id": "weighting", "scheme": "EqualWeighted",
                      "params": {}, "max_weight": None},
        "treatment": {"corporate_actions": "ADJUST_DIVISOR"},
    },
}


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A client holding the index and data, and no backtest."""
    config = ServerConfig(auth_token=TOKEN,
                          data_fetcher=build_fetcher(),
                          storage_root=tmp_path_factory.mktemp("level"))
    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        created = entered.post("/indices", json=copy.deepcopy(INDEX), headers=auth())
        assert created.status_code == 200, created.text
        yield entered


@pytest.fixture(scope="module")
def level(client) -> dict:
    """The index's level over the whole of the data."""
    response = client.get("/indices/EQ2/level", headers=auth())
    assert response.status_code == 200, response.text

    return response.json()


def series(payload: dict) -> pd.Series:
    """A SeriesPayload as a Series."""
    return pd.Series(payload["data"], index=pd.to_datetime(payload["index"]))


class TestTheLevel:

    def test_it_needs_no_backtest(self,
                                  client,
                                  level):
        assert client.get("/strategies", headers=auth()).json()["strategies"] == []
        assert level["index_id"] == "EQ2"
        assert level["name"] == "Equal two"
        assert level["currency"] == "USD"

    def test_it_runs_from_the_base_date_to_the_end_of_the_data(self,
                                                               level):
        levels = series(level["level"])

        assert level["start"] == START and level["end"] == LAST_SESSION
        assert str(levels.index[0].date()) == START
        assert levels.iloc[0] == pytest.approx(1000.0)

    def test_it_is_the_calculation_a_backtest_starts_from(self,
                                                          level):
        """The library's own calculation of the stored definition, through
        the same entry point a run uses."""
        definition = build_definition(IndexDocument.model_validate(INDEX), None)
        expected = Backtest(initial_capital=1.0, data_provider=build_fetcher(),
                            cache=False).calculate(definition, start=START, end=END)

        pd.testing.assert_series_equal(series(level["level"]),
                                       expected.index_levels.rename(None),
                                       check_freq=False, check_names=False)

    def test_it_carries_the_weights_at_each_rebalance(self,
                                                      level):
        rebalances = level["rebalances"]

        assert [snapshot["date"] for snapshot in rebalances] == sorted(
            snapshot["date"] for snapshot in rebalances)
        assert rebalances[0]["weights"] == pytest.approx({"AAA": 0.5, "BBB": 0.5})


class TestItsFigures:

    def test_the_headline_figures_are_the_levels(self,
                                                 level):
        levels = series(level["level"])
        metrics = level["metrics"]

        assert metrics["total_return"] == pytest.approx(levels.iloc[-1] / levels.iloc[0] - 1)
        assert metrics["max_drawdown"] == pytest.approx((levels / levels.cummax() - 1).min())
        assert metrics["volatility"] > 0

    def test_the_tracking_and_fund_figures_are_null(self,
                                                    level):
        metrics = level["metrics"]

        assert metrics["tracking_error"] is None
        assert metrics["money_weighted_return"] is None


class TestTheWindow:

    def test_a_window_is_calculated_from_its_start(self,
                                                   client):
        window = client.get("/indices/EQ2/level", params={"start": "2024-01-02",
                                                           "end": "2024-02-29"},
                            headers=auth()).json()

        assert window["start"] == "2024-01-02"
        assert window["end"] == "2024-02-29"

    def test_an_optimised_index_calculates_through_its_source(self,
                                                              client):
        derived = client.post("/indices/EQ2/optimise",
                              json={"id": "EQ2_OPT", "name": "Optimised",
                                    "constraints": [{"id": "invested", "type": "FullInvestment",
                                                     "params": {"target": 1.0}}]},
                              headers=auth())
        assert derived.status_code == 200, derived.text

        response = client.get("/indices/EQ2_OPT/level", headers=auth())

        assert response.status_code == 200, response.text
        assert response.json()["rebalances"]


class TestRefusals:

    def test_an_unknown_index_is_not_found(self,
                                           client):
        assert client.get("/indices/NOPE/level", headers=auth()).status_code == 404

    def test_without_data_it_answers_no_data_loaded(self,
                                                    tmp_path):
        config = ServerConfig(auth_token=TOKEN, storage_root=tmp_path)
        bare = TestClient(create_app(config), raise_server_exceptions=False)
        bare.post("/indices", json=copy.deepcopy(INDEX), headers=auth())

        response = bare.get("/indices/EQ2/level", headers=auth())

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "NO_DATA_LOADED"
