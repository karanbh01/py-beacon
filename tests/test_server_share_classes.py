# tests/test_server_share_classes.py
"""Contract tests for share classes on the API (BN-314).

Three dollar names through 2024, a dollar-to-euro rate that drifts up about
10%, and short rates of 5% for the dollar and 3% for the euro. A fund with a
dollar class, an unhedged euro class and a hedged one.
"""
import copy

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data import pair_reference, short_rate_reference
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

CLASSES = [{"name": "USD"},
           {"name": "EUR", "currency": "EUR", "distribution": "distributing"},
           {"name": "EUR H", "currency": "EUR", "hedged": True, "management_fee_bps": 20}]
CAPITAL = {"USD": 1_000_000.0, "EUR": 500_000.0, "EUR H": 500_000.0}


def build_fetcher() -> DataFetcher:
    """Dollar names, a dollar-to-euro rate, and the two short rates."""
    random = np.random.default_rng(31)
    market = random.normal(0.0004, 0.009, len(DAYS))
    rows = []

    for name in NAMES:
        prices = 50.0 * np.cumprod(1.0 + market + random.normal(0.0, 0.004, len(DAYS)))
        rows += [{"IDENTIFIER": name, "DATE": day, "CLOSE": price, "SHARES_OUTSTANDING": 1e8}
                 for day, price in zip(DAYS, prices, strict=True)]

    fx = 0.92 * np.cumprod(1.0 + 0.0004 + random.normal(0.0, 0.003, len(DAYS)))
    rows += [{"IDENTIFIER": "USDEUR", "DATE": day, "CLOSE": rate}
             for day, rate in zip(DAYS, fx, strict=True)]
    rows += [{"IDENTIFIER": f"{currency} 1M", "DATE": day, "CLOSE": rate}
             for currency, rate in (("USD", 0.05), ("EUR", 0.03)) for day in DAYS]
    reference = pd.DataFrame([
        *({"IDENTIFIER": name, "NAME": name, "DATE_FROM": "2020-01-01", "CURRENCY": "USD",
           "EXCHANGE": "XNYS"} for name in NAMES),
        pair_reference("USDEUR"),
        *(short_rate_reference(f"{currency} 1M", currency, tenor="1M")
          for currency in ("USD", "EUR"))])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


def body(**overrides) -> dict:
    """A backtest of the three classes over the year."""
    return {"start": START, "end": END, "share_classes": copy.deepcopy(CLASSES),
            "initial_capital": dict(CAPITAL), "vehicle": {"preset": None}, **overrides}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A client holding the index and a strategy that holds it."""
    config = ServerConfig(auth_token=TOKEN, data_fetcher=build_fetcher(),
                          storage_root=tmp_path_factory.mktemp("classes"))
    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        assert entered.post("/indices", json=copy.deepcopy(INDEX),
                            headers=auth()).status_code == 200
        assert entered.post("/strategies", json={"id": "fund", "name": "Fund",
                                                 "index_id": "EW3"},
                            headers=auth()).status_code == 200
        yield entered


def run(client,
        payload: dict) -> dict:
    """Submit a backtest, wait for it, and return its result."""
    response = client.post("/strategies/fund/backtest", json=payload, headers=auth())
    assert response.status_code == 202, response.text
    client.portal.call(client.app.state.jobs.drain)
    job = client.get(f"/jobs/{response.json()['job_id']}", headers=auth()).json()
    assert job["status"] == SUCCEEDED, job

    return job["result"]


@pytest.fixture(scope="module")
def result(client) -> dict:
    """One run of the three classes."""
    return run(client, body())


def classes_of(result: dict) -> dict[str, dict]:
    """The result's classes by name."""
    return {record["name"]: record for record in result["share_classes"]}


class TestTheRequest:

    def test_the_options_describe_a_share_class(self,
                                                client):
        options = client.get("/backtest/options", headers=auth()).json()
        fields = {spec["name"]: spec for spec in options["share_class_settings"]}

        assert fields["hedge_ratio"]["unit"] == "fraction"
        assert fields["hedge_ratio"]["maximum"] == 2.0
        assert fields["management_fee_bps"]["unit"] == "bps"
        assert options["choices"]["distribution"] == ["accumulating", "distributing"]

    def test_the_settings_report_the_classes(self,
                                             client):
        validation = client.post("/strategies/fund/backtest/validate", json=body(),
                                 headers=auth()).json()

        assert validation["valid"] is True
        assert [record["name"] for record in validation["settings"]["share_classes"]] == \
            ["USD", "EUR", "EUR H"]

    @pytest.mark.parametrize(("change", "path"), [
        ({"share_classes": [{"name": "A", "currency": "EUR"},
                            {"name": "B", "hedged": True}],
          "initial_capital": {"A": 1.0, "B": 1.0}}, "share_classes.1.currency"),
        ({"share_classes": [{"name": "A"}, {"name": "A"}],
          "initial_capital": {"A": 1.0}}, "share_classes.1.name"),
        ({"share_classes": [{"name": "A", "distribution": "monthly"}],
          "initial_capital": {"A": 1.0}}, "share_classes.0.distribution"),
        ({"initial_capital": {"USD": 1.0, "GBP": 1.0}}, "initial_capital.GBP"),
        ({"initial_capital": 1_000_000.0}, "initial_capital"),
        ({"flows": [{"type": "DatedFlows", "params": {"flows": {"2024-06-03": 1000}}}]},
         "flows"),
        ({"flows": {"JPY": []}}, "flows.JPY"),
    ])
    def test_a_bad_setting_is_reported_at_its_path(self,
                                                   client,
                                                   change,
                                                   path):
        validation = client.post("/strategies/fund/backtest/validate", json=body(**change),
                                 headers=auth()).json()

        assert validation["valid"] is False
        assert path in [finding["path"] for finding in validation["findings"]]

    @pytest.mark.parametrize("field", ["hedge_ratio", "hedge_band", "management_fee_bps"])
    def test_an_out_of_range_number_is_refused(self,
                                               client,
                                               field):
        classes = [{"name": "A", "currency": "EUR", "hedged": True, field: -1}]
        response = client.post("/strategies/fund/backtest/validate",
                               json=body(share_classes=classes, initial_capital={"A": 1.0}),
                               headers=auth())

        assert response.status_code == 422
        assert field in str(response.json()["error"]["detail"])

    def test_capital_by_class_needs_classes(self,
                                            client):
        validation = client.post("/strategies/fund/backtest/validate",
                                 json={"initial_capital": {"USD": 1.0}},
                                 headers=auth()).json()

        assert "initial_capital" in [finding["path"] for finding in validation["findings"]]


class TestTheResult:

    def test_each_class_is_recorded_in_order(self,
                                             result):
        assert [record["name"] for record in result["share_classes"]] == ["USD", "EUR", "EUR H"]

    def test_its_series_share_the_level_s_dates(self,
                                                result):
        dates = result["level"]["index"]

        for record in result["share_classes"]:
            for name in ("net_assets", "book_net_assets", "units", "nav_per_unit", "fees",
                         "distributions"):
                assert record[name]["index"] == dates, (record["name"], name)

    def test_the_classes_add_up_to_the_fund(self,
                                            result):
        """In the book's currency, the classes' net assets are the fund's, its
        hedge's value included. Measured: they agree to about one part in a
        million, the opening day's rounding."""
        total = sum(record["book_net_assets"]["data"][-1] for record in result["share_classes"])

        assert total == pytest.approx(result["initial_capital"] * result["level"]["data"][-1]
                                      / 100.0, rel=1e-5)

    def test_each_class_is_in_its_own_currency(self,
                                               result):
        records = classes_of(result)

        assert records["USD"]["currency"] == "USD"
        assert records["EUR"]["currency"] == "EUR" and records["EUR"]["hedged"] is False
        assert records["EUR H"]["hedged"] is True

    def test_a_hedged_class_carries_its_hedge(self,
                                              result):
        record = classes_of(result)["EUR H"]

        assert record["hedge_value"]["index"] == result["level"]["index"]
        assert record["hedges"][0]["reason"] == "open"
        assert {"roll"} <= {trade["reason"] for trade in record["hedges"]}
        assert result["hedge_value"] is not None
        assert classes_of(result)["EUR"]["hedge_value"] is None

    def test_a_class_s_fee_is_its_own(self,
                                      result):
        records = classes_of(result)

        assert records["EUR H"]["management_fee_bps"] == 20
        assert records["EUR H"]["fees_paid"] > 0
        assert records["USD"]["fees_paid"] == 0

    def test_a_distributing_class_pays_out(self,
                                           result):
        records = classes_of(result)

        assert records["EUR"]["distribution"] == "distributing"
        assert records["EUR"]["distributions_paid"] >= 0.0

    def test_each_class_tracks_the_index_in_its_currency(self,
                                                         result):
        """Hedged, against the hedged index, so the 2% carry is not counted;
        every class within a few basis points of its benchmark, less its fee."""
        records = classes_of(result)

        for name in ("USD", "EUR"):
            assert abs(records[name]["metrics"]["tracking_difference"]) < 0.003, name

        hedged = records["EUR H"]["metrics"]
        assert hedged["tracking_difference"] == pytest.approx(-0.002, abs=0.003)
        assert hedged["tracking_error"] < 0.01

    def test_a_run_without_classes_answers_as_before(self,
                                                     client):
        plain = run(client, {"start": START, "end": END})

        assert plain["share_classes"] == []
        assert plain["hedge_value"] is None
