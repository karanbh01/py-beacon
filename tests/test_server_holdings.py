# tests/test_server_holdings.py
"""BN-326: a strategy's holdings workflow at a date, from the universe to the
units held.

Five names through a quarterly-rebalanced backtest over the first half of
2024. The index keeps names worth at least $1bn (EEE is too small) and caps a
weight at 30% (AAA, the largest, is cut). The strategy's implementation then
drops names under $20 (DDD), excludes CCC outright, keeps 2% in cash and caps
a position at 45%.
"""
import copy

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.server import ServerConfig, create_app
from beacon.server.jobs import SUCCEEDED

TOKEN = "test-token-value"
START, END = "2024-01-02", "2024-06-28"
DAYS = pd.bdate_range(START, END)
# Price and shares: market caps of $5bn, $3bn, $2bn, $1.5bn and $0.2bn.
NAMES = {"AAA": (100.0, 50e6), "BBB": (60.0, 50e6), "CCC": (40.0, 50e6),
         "DDD": (10.0, 150e6), "EEE": (25.0, 8e6)}
MID = "2024-02-15"

INDEX = {
    "id": "CAP5",
    "name": "Capped five",
    "base_date": START,
    "base_value": 1000.0,
    "currency": "USD",
    "calendar": "XNYS",
    "rebalancing_frequency": "QUARTERLY",
    "universe": {"universe_id": None, "identifiers": list(NAMES)},
    "pipeline": {
        "selection": [{"id": "big-enough", "type": "MarketCapRule",
                       "params": {"min_market_cap": 1e9}}],
        "weighting": {"id": "weighting", "scheme": "MarketCapWeighted", "params": {},
                      "max_weight": 0.3},
        "treatment": {"corporate_actions": "ADJUST_DIVISOR"},
    },
}
IMPLEMENTATION = {
    "screens": [{"type": "MinimumPriceScreen", "params": {"min_price": 20.0}},
                {"type": "ExclusionScreen", "params": {"identifiers": ["CCC"]}}],
    "caps": [{"type": "WeightCap", "params": {"max_weight": 0.45}}],
    "cash_buffer": 0.02,
}


def build_fetcher() -> DataFetcher:
    """Five names drifting apart a little each day."""
    random = np.random.default_rng(43)
    rows = []

    for name, (price, shares) in NAMES.items():
        path = price * np.cumprod(1.0 + random.normal(0.0003, 0.008, len(DAYS)))
        rows += [{"IDENTIFIER": name, "DATE": day, "CLOSE": close, "SHARES_OUTSTANDING": shares,
                  "VOLUME": 1e6} for day, close in zip(DAYS, path, strict=True)]

    reference = pd.DataFrame([{"IDENTIFIER": name, "NAME": f"Name {name}", "CURRENCY": "USD",
                               "EXCHANGE": "NYSE", "DATE_FROM": "2020-01-01",
                               "SECTOR": "Industrials"} for name in NAMES])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """The index and the strategy, backtested once over the half year."""
    config = ServerConfig(auth_token=TOKEN, data_fetcher=build_fetcher(),
                          storage_root=tmp_path_factory.mktemp("holdings"))

    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        assert entered.post("/indices", json=copy.deepcopy(INDEX),
                            headers=auth()).status_code == 200
        assert entered.post("/strategies", json={
            "id": "fund", "name": "Fund", "index_id": "CAP5",
            "implementation": IMPLEMENTATION}, headers=auth()).status_code == 200
        response = entered.post("/strategies/fund/backtest",
                                json={"start": START, "end": END, "initial_capital": 1e6},
                                headers=auth())
        assert response.status_code == 202, response.text
        entered.portal.call(entered.app.state.jobs.drain)
        job = entered.get(f"/jobs/{response.json()['job_id']}", headers=auth()).json()
        assert job["status"] == SUCCEEDED, job
        yield entered


def holdings(client,
             **params) -> dict:
    """The holdings view."""
    response = client.get("/strategies/fund/holdings", params=params, headers=auth())
    assert response.status_code == 200, response.text

    return response.json()


def rows_of(view: dict) -> dict[str, dict]:
    """The view's rows by name."""
    return {row["identifier"]: row for row in view["rows"]}


@pytest.fixture(scope="module")
def first(client) -> dict:
    """The workflow on the first rebalance."""
    return holdings(client, date=START)


class TestTheDates:

    def test_it_lists_every_rebalance_and_the_one_in_force(self,
                                                           client):
        view = holdings(client, date=MID)

        assert view["rebalance_dates"][0] == START
        assert view["rebalance_date"] == START and view["as_of"] == MID

    def test_without_a_date_it_reads_the_run_s_last_day(self,
                                                        client):
        view = holdings(client)

        assert view["as_of"] == END
        assert view["rebalance_date"] == view["rebalance_dates"][-1]

    def test_a_date_before_the_first_rebalance_is_not_found(self,
                                                            client):
        response = client.get("/strategies/fund/holdings", params={"date": "2023-06-30"},
                              headers=auth())

        assert response.status_code == 404


class TestTheStages:

    def test_they_run_from_the_universe_to_the_caps(self,
                                                    first):
        assert [(stage["key"], stage["kind"]) for stage in first["stages"]] == [
            ("universe", "universe"),
            ("big-enough", "index_rule"),
            ("index_cap", "index_cap"),
            ("stale price", "implementation_screen"),
            ("MinimumPriceScreen", "implementation_screen"),
            ("ExclusionScreen", "implementation_screen"),
            ("cash_buffer", "cash_buffer"),
            ("implementation_cap", "implementation_cap")]

    def test_each_says_what_a_column_reads_and_what_is_left(self,
                                                            first):
        stages = {stage["key"]: stage for stage in first["stages"]}

        assert stages["big-enough"]["label"] == "Market capitalisation"
        assert stages["MinimumPriceScreen"]["label"] == "Minimum price"
        assert [stage["remaining"] for stage in first["stages"]] == [5, 4, 4, 4, 3, 2, 2, 2]


class TestTheRows:

    def test_every_name_in_the_universe_has_a_row(self,
                                                  first):
        assert set(rows_of(first)) == set(NAMES)

    def test_a_name_the_index_excluded_says_where(self,
                                                  first):
        tiny = rows_of(first)["EEE"]

        assert tiny["excluded_at"] == "big-enough"
        assert tiny["index_weight"] is None and tiny["portfolio_weight"] is None

    def test_the_index_s_cap_is_shown_before_and_after(self,
                                                       first):
        largest = rows_of(first)["AAA"]

        assert largest["index_weight"] == pytest.approx(0.3)
        assert largest["index_weight_uncapped"] > 0.3

    def test_a_name_an_implementation_screen_removed_says_which(self,
                                                                first):
        rows = rows_of(first)

        assert rows["DDD"]["excluded_at"] == "MinimumPriceScreen"
        assert rows["DDD"]["implementation"]["MinimumPriceScreen"] is None
        assert rows["CCC"]["excluded_at"] == "ExclusionScreen"
        assert rows["CCC"]["index_weight"] is not None

    def test_a_kept_name_carries_its_weight_after_each_stage(self,
                                                             first):
        stages = rows_of(first)["AAA"]["implementation"]

        assert stages["cash_buffer"] == pytest.approx(stages["ExclusionScreen"] * 0.98)
        assert stages["implementation_cap"] == pytest.approx(0.45)

    def test_the_portfolio_and_cash_add_up_to_the_book(self,
                                                       first):
        invested = sum(row["portfolio_weight"] or 0.0 for row in first["rows"])

        assert invested + first["cash_weight"] == pytest.approx(1.0)

    def test_the_reference_columns_are_on_the_rows(self,
                                                   client):
        view = holdings(client, date=START, reference_fields="name,sector")

        assert rows_of(view)["AAA"]["reference"] == {"name": "Name AAA",
                                                     "sector": "Industrials"}


class TestWhatWasHeld:

    def test_on_the_rebalance_date_there_is_no_drift(self,
                                                     first):
        largest = rows_of(first)["AAA"]

        assert largest["weight_drift"] is None
        assert largest["shares_held"] > 0

    def test_between_rebalances_the_drift_is_held_less_target(self,
                                                              client):
        largest = rows_of(holdings(client, date=MID))["AAA"]

        assert largest["weight_drift"] == pytest.approx(
            largest["current_weight"] - largest["portfolio_weight"])

    def test_the_held_weight_is_the_book_s_own(self,
                                               client):
        """The units at the day's price over the NAV: what the book recorded."""
        record = client.get("/strategies/fund/record", params={"reference_fields": ""},
                            headers=auth()).json()
        panel = record["portfolio"]["weights"]
        day = next(label for label in panel["index"] if label.startswith(MID))
        recorded = dict(zip(panel["columns"], panel["data"][panel["index"].index(day)],
                            strict=True))

        held = rows_of(holdings(client, date=MID))

        for name in ("AAA", "BBB"):
            assert held[name]["current_weight"] == pytest.approx(recorded[name], rel=1e-9)

    def test_a_name_never_held_has_no_units(self,
                                            first):
        assert rows_of(first)["DDD"]["shares_held"] is None


class TestOtherRuns:

    def test_a_benchmark_index_s_weights_are_given(self,
                                                   client):
        """Benchmarked to an equal-weighted index over the same names."""
        equal = copy.deepcopy(INDEX)
        equal.update(id="EQ5", name="Equal five")
        equal["pipeline"]["selection"] = []
        equal["pipeline"]["weighting"] = {"id": "weighting", "scheme": "EqualWeighted",
                                          "params": {}, "max_weight": None}
        assert client.post("/indices", json=equal, headers=auth()).status_code == 200
        response = client.post("/strategies/fund/backtest",
                               json={"start": START, "end": END,
                                     "benchmark": {"kind": "index", "id": "EQ5"}},
                               headers=auth())
        assert response.status_code == 202, response.text
        client.portal.call(client.app.state.jobs.drain)

        rows = rows_of(holdings(client, date=START))

        assert rows["EEE"]["benchmark_weight"] == pytest.approx(0.2)
        assert rows["EEE"]["index_weight"] is None

    def test_without_an_implementation_the_index_is_traded_to(self,
                                                              client):
        assert client.post("/strategies", json={"id": "plain", "name": "Plain",
                                                "index_id": "CAP5"},
                           headers=auth()).status_code == 200
        response = client.post("/strategies/plain/backtest",
                               json={"start": START, "end": END}, headers=auth())
        assert response.status_code == 202
        client.portal.call(client.app.state.jobs.drain)

        view = client.get("/strategies/plain/holdings", params={"date": START},
                          headers=auth()).json()

        # Stale prices are screened whatever the implementation; with nothing
        # to cap, there is no capacity stage.
        assert [stage["key"] for stage in view["stages"]] == ["universe", "big-enough",
                                                              "index_cap", "stale price"]
        largest = rows_of(view)["AAA"]
        assert largest["portfolio_weight"] == pytest.approx(largest["index_weight"])
        assert largest["implementation"] == {"stale price": pytest.approx(0.3)}

    def test_a_run_saved_before_this_still_answers(self,
                                                   client):
        """Without the saved stages, the index's weights and the book still show."""
        client.app.state.holdings_store.delete("plain")

        view = client.get("/strategies/plain/holdings", params={"date": START},
                          headers=auth()).json()

        assert [(stage["key"], stage["remaining"]) for stage in view["stages"]] == [
            ("index_cap", 4)]
        assert rows_of(view)["AAA"]["shares_held"] is None
        assert rows_of(view)["AAA"]["index_weight"] == pytest.approx(0.3)


class TestUniverseWeights:
    """BN-329: each universe name's share of the universe's market cap."""

    def test_every_universe_name_has_its_share(self,
                                               first):
        rows = rows_of(first)
        total_cap = sum(price * shares for price, shares in NAMES.values())

        assert sum(row["universe_weight"] for row in rows.values()) == pytest.approx(1.0)
        assert rows["AAA"]["universe_weight"] == pytest.approx(5e9 / total_cap, abs=0.01)
        assert rows["EEE"]["universe_weight"] == pytest.approx(0.2e9 / total_cap, abs=0.005)

    def test_the_preview_carries_them_too(self,
                                          client):
        response = client.post("/indices/CAP5/preview", json={"as_of": START}, headers=auth())
        assert response.status_code == 200, response.text

        assets = {asset["identifier"]: asset for asset in response.json()["assets"]}
        assert sum(asset["universe_weight"] for asset in assets.values()) == pytest.approx(1.0)
        assert assets["EEE"]["universe_weight"] > 0


class TestStageDescriptions:
    """BN-328: what each stage is and does, for a column's hover."""

    def test_each_stage_says_what_it_does_with_its_settings(self,
                                                            first):
        stages = {stage["key"]: stage for stage in first["stages"]}

        assert stages["universe"]["description"] == "Starts from the 5 names in its list."
        assert stages["big-enough"]["type"] == "MarketCapRule"
        assert stages["big-enough"]["description"] == (
            "Keeps names whose market cap is at least 1,000,000,000 USD.")
        assert stages["big-enough"]["params"] == [
            {"label": "Minimum market cap", "value": "1,000,000,000 USD"}]
        assert stages["index_cap"]["description"] == (
            "Caps each name at 30% of the index and spreads the excess across the rest.")
        assert stages["MinimumPriceScreen"]["description"] == "Drops names priced below 20 USD."
        assert stages["ExclusionScreen"]["description"] == "Drops CCC."
        assert stages["cash_buffer"]["params"] == [{"label": "Cash buffer", "value": "2%"}]
        assert stages["implementation_cap"]["description"] == (
            "Cuts each position to at most 45% of the book and spreads what it moves across "
            "the rest.")

    def test_the_preview_s_steps_say_so_too(self,
                                            client):
        steps = client.post("/indices/CAP5/preview", json={"as_of": START},
                            headers=auth()).json()["steps"]

        assert steps[0]["type"] == "Universe"
        assert steps[1]["type"] == "MarketCapRule"
        assert steps[1]["description"] == (
            "Keeps names whose market cap is at least 1,000,000,000 USD.")
        assert steps[1]["params"][0]["label"] == "Minimum market cap"
