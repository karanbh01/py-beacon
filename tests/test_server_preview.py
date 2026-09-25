# tests/test_server_preview.py
"""Contract tests for the constituent preview endpoint (derivation waterfall)."""
import copy
from itertools import pairwise

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.index.calculation import IndexCalculator
from beacon.server import ServerConfig, create_app
from beacon.server.definitions import build_index_definition
from beacon.server.preview import build_preview
from beacon.server.schemas import IndexDocument
from beacon.server.store import DocumentStore

TOKEN = "test-token-value"
AS_OF = "2025-01-02"
DATES = pd.bdate_range("2024-11-01", "2025-01-31")

# Six names with deliberately spread market caps and volumes, so the rules bite
# in a known way: shares are identical, price drives the cap.
#   AAA 500  BBB 300  CCC 100  DDD 60  EEE 30  FFF 10   (price x 1_000 shares)
PRICES = {"AAA": 500.0, "BBB": 300.0, "CCC": 100.0,
          "DDD": 60.0, "EEE": 30.0, "FFF": 10.0}
VOLUMES = {"AAA": 900_000, "BBB": 800_000, "CCC": 700_000,
           "DDD": 600_000, "EEE": 500_000, "FFF": 1_000}
SHARES = 1_000


def build_fetcher() -> DataFetcher:
    """Synthetic market and reference data for the six names."""
    market = pd.DataFrame([
        {"IDENTIFIER": name,
         "DATE": date,
         "CLOSE": price,
         "VOLUME": VOLUMES[name],
         "SHARES_OUTSTANDING": SHARES}
        for name, price in PRICES.items()
        for date in DATES
    ])
    reference = pd.DataFrame([
        {"IDENTIFIER": name,
         "DATE_FROM": "2020-01-01",
         "NAME": name,
         "CURRENCY": "USD",
         "EXCHANGE": "NYSE"}
        for name in PRICES
    ])

    return DataFetcher(MarketData.from_dataframe(market),
                       ReferenceData.from_dataframe(reference))


def definition_document(**overrides) -> dict:
    """An index whose market-cap rule excludes the two smallest names."""
    document = {
        "id": "PREVIEW",
        "name": "Preview Index",
        "base_date": AS_OF,
        "base_value": 1000.0,
        "currency": "USD",
        "calendar": "XNYS",
        "rebalancing_frequency": "QUARTERLY",
        "description": None,
        "universe": {"universe_id": None, "identifiers": list(PRICES)},
        "pipeline": {
            "selection": [
                # Cuts EEE (30k) and FFF (10k); keeps AAA..DDD.
                {"id": "min-mcap",
                 "type": "MarketCapRule",
                 "params": {"min_market_cap": 50_000.0}},
                # Cuts FFF on volume, but the cap rule already removed it —
                # so FFF must be attributed to min-mcap, not to this rule.
                {"id": "liquidity",
                 "type": "LiquidityRule",
                 "params": {"min_avg_daily_volume": 100_000, "lookback_days": 30}},
            ],
            "weighting": {"id": "weighting",
                          "scheme": "MarketCapWeighted",
                          "params": {},
                          "max_weight": None},
            "treatment": {"corporate_actions": "ADJUST_DIVISOR"},
        },
    }
    document.update(overrides)

    return copy.deepcopy(document)


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(tmp_path) -> TestClient:
    """Client with a data source and isolated storage, holding the index."""
    config = ServerConfig(auth_token=TOKEN,
                          data_fetcher=build_fetcher(),
                          storage_root=tmp_path)
    client = TestClient(create_app(config), raise_server_exceptions=False)
    created = client.post("/indices", json=definition_document(), headers=auth())
    assert created.status_code == 200, created.json()

    return client


def preview(client,
            **body) -> dict:
    """Run the preview endpoint and return its payload."""
    response = client.post("/indices/PREVIEW/preview", json=body or {}, headers=auth())
    assert response.status_code == 200, response.json()

    return response.json()


class TestWaterfall:

    def test_first_step_is_the_universe(self,
                                        client):
        steps = preview(client)["steps"]

        assert steps[0]["position"] == 0
        assert steps[0]["rule_id"] is None
        assert steps[0]["remaining"] == len(PRICES)

    def test_one_step_per_rule_in_order(self,
                                        client):
        steps = preview(client)["steps"]

        assert [s["rule_id"] for s in steps] == [None, "min-mcap", "liquidity"]
        assert [s["position"] for s in steps] == [0, 1, 2]

    def test_counts_are_monotonically_non_increasing(self,
                                                     client):
        """The acceptance criterion: the funnel never widens."""
        remaining = [step["remaining"] for step in preview(client)["steps"]]

        assert all(later <= earlier for earlier, later in pairwise(remaining))

    def test_each_step_lists_what_it_removed(self,
                                             client):
        steps = preview(client)["steps"]
        market_cap_step = steps[1]

        assert market_cap_step["excluded"] == ["EEE", "FFF"]
        assert market_cap_step["remaining"] == 4

    def test_removal_counts_reconcile_with_remaining(self,
                                                     client):
        steps = preview(client)["steps"]

        for earlier, later in pairwise(steps):
            assert later["remaining"] == earlier["remaining"] - len(later["excluded"])


class TestAttribution:
    """Every excluded asset reports the rule that excluded it."""

    def test_every_excluded_asset_names_a_rule(self,
                                               client):
        assets = preview(client)["assets"]
        excluded = [a for a in assets if not a["included"]]

        assert excluded, "expected some exclusions in this fixture"
        assert all(a["excluded_by"] is not None for a in excluded)
        assert all(a["excluded_at"] is not None for a in excluded)

    def test_included_assets_name_no_rule(self,
                                          client):
        assets = preview(client)["assets"]

        for asset in assets:
            if asset["included"]:
                assert asset["excluded_by"] is None
                assert asset["excluded_at"] is None

    def test_the_first_rule_to_exclude_owns_the_asset(self,
                                                      client):
        """FFF fails both rules; the earlier one must be the attribution."""
        assets = {a["identifier"]: a for a in preview(client)["assets"]}

        assert assets["FFF"]["excluded_by"] == "min-mcap"
        assert assets["FFF"]["excluded_at"] == 1

    def test_every_universe_member_appears_exactly_once(self,
                                                        client):
        assets = preview(client)["assets"]

        assert sorted(a["identifier"] for a in assets) == sorted(PRICES)

    def test_step_exclusions_match_the_asset_rows(self,
                                                  client):
        payload = preview(client)
        from_steps = {identifier
                      for step in payload["steps"]
                      for identifier in step["excluded"]}
        from_assets = {a["identifier"] for a in payload["assets"] if not a["included"]}

        assert from_steps == from_assets


class TestWeights:

    def test_final_weights_sum_to_one(self,
                                      client):
        """The acceptance criterion, stated there as 100.00%."""
        payload = preview(client)

        assert payload["total_weight"] == pytest.approx(1.0)
        assert sum(payload["weights"].values()) == pytest.approx(1.0)

    def test_weights_cover_exactly_the_survivors(self,
                                                 client):
        payload = preview(client)
        included = {a["identifier"] for a in payload["assets"] if a["included"]}

        assert set(payload["weights"]) == included
        assert included == {"AAA", "BBB", "CCC", "DDD"}

    def test_weights_are_proportional_to_market_cap(self,
                                                    client):
        weights = preview(client)["weights"]

        assert weights["AAA"] / weights["BBB"] == pytest.approx(500 / 300)

    def test_excluded_assets_have_no_weight(self,
                                            client):
        assets = preview(client)["assets"]

        assert all(a["weight"] is None for a in assets if not a["included"])


class TestCapping:
    """Capped names sit exactly at the cap."""

    @pytest.fixture
    def capped_client(self,
                      tmp_path) -> TestClient:
        document = definition_document()
        document["pipeline"]["weighting"]["max_weight"] = 0.4
        config = ServerConfig(auth_token=TOKEN,
                              data_fetcher=build_fetcher(),
                              storage_root=tmp_path)
        client = TestClient(create_app(config), raise_server_exceptions=False)
        created = client.post("/indices", json=document, headers=auth())
        assert created.status_code == 200, created.json()

        return client

    def test_capped_name_sits_exactly_at_the_cap(self,
                                                 capped_client):
        payload = preview(capped_client)

        # AAA is 500/960 uncapped, comfortably over a 40% cap.
        assert payload["weights"]["AAA"] == pytest.approx(0.4)

    def test_capped_flag_and_uncapped_weight_are_reported(self,
                                                          capped_client):
        assets = {a["identifier"]: a for a in preview(capped_client)["assets"]}

        assert assets["AAA"]["capped"] is True
        assert assets["AAA"]["uncapped_weight"] == pytest.approx(500 / 960)
        assert assets["BBB"]["capped"] is False
        assert assets["BBB"]["uncapped_weight"] is None

    def test_weights_still_sum_to_one_after_capping(self,
                                                    capped_client):
        assert preview(capped_client)["total_weight"] == pytest.approx(1.0)

    def test_no_weight_exceeds_the_cap(self,
                                       capped_client):
        payload = preview(capped_client)

        assert max(payload["weights"].values()) <= 0.4 + 1e-9
        assert payload["cap"] == 0.4

    def test_redistributed_amount_is_reported(self,
                                              capped_client):
        payload = preview(capped_client)

        assert payload["cap_redistributed"] == pytest.approx(500 / 960 - 0.4)

    def test_uncapped_index_reports_no_cap(self,
                                           client):
        payload = preview(client)

        assert payload["cap"] is None
        assert payload["cap_redistributed"] == 0.0


class TestConsistencyWithTheCalculator:
    """Preview must not drift from what a real run would select."""

    def test_survivors_match_select_constituents(self,
                                                 tmp_path):
        document = IndexDocument.model_validate(definition_document())
        fetcher = build_fetcher()
        date = pd.Timestamp(AS_OF)

        payload = build_preview(document, fetcher,
                                DocumentStore("indices", tmp_path), AS_OF)
        from_preview = {a.identifier for a in payload.assets if a.included}

        calculator = IndexCalculator(build_index_definition(document), fetcher)
        universe = calculator.resolve_universe(date)
        from_calculator = {a.asset_id
                           for a in calculator.select_constituents(universe, date)}

        assert from_preview == from_calculator

    def test_weights_match_the_calculator(self,
                                          tmp_path):
        document = IndexDocument.model_validate(definition_document())
        fetcher = build_fetcher()
        date = pd.Timestamp(AS_OF)

        payload = build_preview(document, fetcher,
                                DocumentStore("indices", tmp_path), AS_OF)

        calculator = IndexCalculator(build_index_definition(document), fetcher)
        universe = calculator.resolve_universe(date)
        constituents = calculator.select_constituents(universe, date)
        raw = calculator.calculate_constituent_weights(constituents, date)
        expected, _ = calculator.cap_weights(raw)

        for asset, weight in expected.items():
            assert payload.weights[asset.asset_id] == pytest.approx(weight)


class TestEndpointBehaviour:

    def test_defaults_to_the_base_date(self,
                                       client):
        assert preview(client)["as_of"] == AS_OF

    def test_honours_an_explicit_as_of(self,
                                       client):
        assert preview(client, as_of="2025-01-15")["as_of"] == "2025-01-15"

    def test_unknown_index_is_404(self,
                                  client):
        response = client.post("/indices/ABSENT/preview", json={}, headers=auth())

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "DATA_NOT_FOUND"

    def test_requires_authentication(self,
                                     client):
        assert client.post("/indices/PREVIEW/preview", json={}).status_code == 401

    def test_without_data_is_refused_until_a_store_is_loaded(self,
                                                             tmp_path):
        """Preview evaluates real rules, so it cannot run without data."""
        config = ServerConfig(auth_token=TOKEN, storage_root=tmp_path)
        client = TestClient(create_app(config), raise_server_exceptions=False)
        client.post("/indices", json=definition_document(), headers=auth())

        response = client.post("/indices/PREVIEW/preview", json={}, headers=auth())

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "NO_DATA_LOADED"

    def test_documented_in_openapi(self,
                                   client):
        paths = client.app.openapi()["paths"]

        assert "/indices/{index_id}/preview" in paths


# ---------------------------------------------------------------------------
# BN-179: request and resolution are two dates, and only one was published
# BN-182: and the rules resolve the same way, so this runs on the real index
# ---------------------------------------------------------------------------
#
# These tests used a rule-free copy of the index when BN-179 landed, because the
# eligibility rules still read prices on the exact date asked for and so
# excluded every name on a closed day before the weighting was ever reached.
# BN-182 put `MarketCapRule` on the same session primitive, so the isolation is
# no longer needed — and running them against the rule-carrying document is now
# the point, since it is what proves the two layers resolve a shut market
# identically.

EXPECTED_CONSTITUENTS = 4  # AAA..DDD; min-mcap cuts EEE and FFF.


class TestRequestAndResolution:
    """`as_of` is what was asked for; `resolved_date` is what was read.

    The two coincide on a trading day, which is why nothing revealed the
    difference until a preview was asked for a day the market was shut.
    """

    def test_a_trading_day_resolves_to_itself(self,
                                              client):
        payload = preview(client, as_of="2025-01-02")

        assert payload["as_of"] == "2025-01-02"
        assert payload["resolved_date"] == "2025-01-02"

    def test_a_weekend_restates_the_date_asked_for(self,
                                                   client):
        assert preview(client, as_of="2025-01-04")["as_of"] == "2025-01-04"

    def test_a_weekend_names_the_session_it_read(self,
                                                 client):
        """Saturday the 4th; the market last traded on Friday the 3rd."""
        payload = preview(client, as_of="2025-01-04")

        assert payload["resolved_date"] == "2025-01-03"

    def test_a_weekend_still_selects_a_universe(self,
                                                client):
        """BN-182's case: the rules used to exclude every name on a Saturday."""
        weights = preview(client, as_of="2025-01-04")["weights"]

        assert len(weights) == EXPECTED_CONSTITUENTS

    def test_a_weekend_runs_the_rules_rather_than_waving_them_through(self,
                                                                      client):
        """Resolution is not a pass-through: the small names are still cut."""
        payload = preview(client, as_of="2025-01-04")
        excluded = {asset["identifier"] for asset in payload["assets"]
                    if not asset["included"]}

        assert excluded == {"EEE", "FFF"}

    def test_a_weekend_weights_by_market_cap_not_equally(self,
                                                         client):
        """The composition in force on Saturday is Friday's, not an equal one."""
        weights = preview(client, as_of="2025-01-04")["weights"]

        assert len({round(weight, 9) for weight in weights.values()}) == len(weights)

    def test_a_weekend_matches_the_session_it_resolved_to(self,
                                                          client):
        saturday = preview(client, as_of="2025-01-04")["weights"]
        friday = preview(client, as_of="2025-01-03")["weights"]

        assert saturday == pytest.approx(friday)

    def test_past_the_last_bar_is_refused_rather_than_backfilled(self,
                                                                 client):
        response = client.post("/indices/PREVIEW/preview",
                               json={"as_of": "2025-03-03"}, headers=auth())

        assert response.status_code == 500
        assert response.json()["error"]["code"] == "CALCULATION_ERROR"

    def test_the_refusal_names_both_dates(self,
                                          client):
        response = client.post("/indices/PREVIEW/preview",
                               json={"as_of": "2025-03-03"}, headers=auth())
        message = response.json()["error"]["message"]

        assert "2025-03-03" in message
        assert "2025-01-31" in message

    def test_it_refuses_rather_than_previewing_an_empty_index(self,
                                                              client):
        """The harm BN-182 names: an index of no constituents still computes."""
        response = client.post("/indices/PREVIEW/preview",
                               json={"as_of": "2025-03-03"}, headers=auth())

        assert "weights" not in response.json()


class TestTheWeightingSchemeMayBeTheFirstToRefuse:
    """BN-196. The two tests above pass on an index that has selection rules.

    A rule refuses first, so the refusal never reaches the weighting scheme and
    never meets the `except Exception` wrapped around it — which since BN-194
    relabelled a scheme's own `CalculationError` as a crash. Strip the rules and
    the same request, same scheme, same date and same store publish
    `UNEXPECTED_CALCULATION_FAILURE` instead, with the remedy demoted to a
    nested string.

    An index with no selection rules is not an edge case: "hold these names,
    weight them by market cap" is the simplest index there is. It was untested
    here only because one fixture serves the whole file and that fixture carries
    rules, so every refusal in it was attributed before the weighting ran.
    """

    @pytest.fixture
    def unfiltered(self,
                   tmp_path) -> TestClient:
        """The same index with its selection stage empty."""
        document = definition_document()
        document["pipeline"]["selection"] = []

        config = ServerConfig(auth_token=TOKEN,
                              data_fetcher=build_fetcher(),
                              storage_root=tmp_path)
        client = TestClient(create_app(config), raise_server_exceptions=False)

        with client:
            client.post("/indices", json=document, headers=auth())
            yield client

    def test_a_scheme_refusal_is_published_as_a_decision(self,
                                                         unfiltered):
        response = unfiltered.post("/indices/PREVIEW/preview",
                                   json={"as_of": "2025-03-03"}, headers=auth())

        assert response.json()["error"]["code"] == "CALCULATION_ERROR"

    def test_the_remedy_is_the_message_rather_than_a_nested_detail(self,
                                                                   unfiltered):
        """What the user reads when the preview fails, and what they act on."""
        response = unfiltered.post("/indices/PREVIEW/preview",
                                   json={"as_of": "2025-03-03"}, headers=auth())
        message = response.json()["error"]["message"]

        assert "unexpected" not in message.lower()
        assert "2025-01-31" in message

    def test_both_shapes_of_the_index_agree(self,
                                            client,
                                            unfiltered):
        """The classification cannot depend on whether a rule refused first."""
        filtered_code = client.post(
            "/indices/PREVIEW/preview",
            json={"as_of": "2025-03-03"}, headers=auth()).json()["error"]["code"]
        unfiltered_code = unfiltered.post(
            "/indices/PREVIEW/preview",
            json={"as_of": "2025-03-03"}, headers=auth()).json()["error"]["code"]

        assert filtered_code == unfiltered_code == "CALCULATION_ERROR"
