# tests/test_server_backtest_settings.py
"""BN-272: the backtest API's strategies, implementation, flows, vehicles and
modelling assumptions.

Reuses test_server_backtest's two-name index and data. Runs go through the
job, as a client's would; the settings are checked through
`/backtest/validate`, which builds them without running.
"""
import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from test_server_backtest import TOKEN, auth, build_fetcher, index_document, run_backtest

UK_OEIC = {"preset": "uk_oeic", "settings": {"management_fee_bps": 15}}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    config = ServerConfig(auth_token=TOKEN, data_fetcher=build_fetcher(),
                          storage_root=tmp_path_factory.mktemp("settings"))

    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        assert entered.post("/indices", json=index_document(),
                            headers=auth()).status_code == 200
        yield entered


def validate(client,
             **body) -> dict:
    response = client.post("/beacon/BT/backtest/validate", json=body, headers=auth())
    assert response.status_code == 200, response.text

    return response.json()


def paths(validation: dict) -> dict[str, str]:
    return {finding["path"]: finding["code"] for finding in validation["findings"]}


@pytest.fixture(scope="module")
def options(client) -> dict:
    response = client.get("/backtest/options", headers=auth())
    assert response.status_code == 200, response.text

    return response.json()


class TestTheOptions:

    def test_every_family_lists_its_types(self,
                                          options):
        names = {family: [spec["name"] for spec in options[family]]
                 for family in ("screens", "caps", "flows", "pricing", "limits",
                                "replications", "signals", "constructions",
                                "active_constraints")}

        assert "SwingPricing" in names["pricing"]
        assert "OptimisedReplication" in names["replications"]
        assert "FunctionSignal" not in names["signals"]
        assert all(names.values())

    def test_fields_carry_their_units_and_bounds(self,
                                                 options):
        swing = next(spec for spec in options["pricing"] if spec["name"] == "SwingPricing")
        fields = {field["name"]: field for field in swing["parameters"]}

        assert fields["factor_bps"]["unit"] == "bps"
        assert (fields["threshold"]["unit"], fields["threshold"]["maximum"]) == (
            "fraction", 1.0)

    def test_presets_start_with_the_generic_vehicle(self,
                                                    options):
        presets = {preset["key"]: preset for preset in options["presets"]}

        assert options["presets"][0]["key"] == "generic"
        assert presets["ucits_etf"]["archetype"] == "exchange_traded"
        assert "market" in presets["ucits_etf"]["settings"]
        assert presets["uk_oeic"]["settings"]["pricing"]["type"] == "SwingPricing"

    def test_exchange_traded_settings_say_so(self,
                                             options):
        applies = {field["name"]: field["applies_to"]
                   for field in options["vehicle_settings"]}

        assert applies["creation_unit"] == ["exchange_traded"]
        assert applies["management_fee_bps"] is None

    def test_the_modelling_assumptions_carry_the_engine_defaults(self,
                                                                 options):
        defaults = {field["name"]: field["default"]
                    for field in options["modelling_assumptions"]}

        assert defaults["periods_per_year"] == 252
        assert defaults["volume_backfill_days"] == 5
        assert defaults["fx_policy"] == "CARRY_FORWARD"

    def test_it_needs_no_data(self,
                              tmp_path):
        bare = TestClient(create_app(ServerConfig(auth_token=TOKEN,
                                                  storage_root=tmp_path)))

        assert bare.get("/backtest/options", headers=auth()).status_code == 200


class TestValidation:

    def test_a_preset_is_filled_in_with_the_changes(self,
                                                    client):
        settings = validate(client, vehicle=UK_OEIC)["settings"]["vehicle"]

        assert settings["preset"] == "uk_oeic"
        assert settings["settings"]["management_fee_bps"] == 15
        assert settings["settings"]["limits"][0]["type"] == "UcitsLimits"

    def test_a_typed_setting_replaces_and_a_plain_one_merges(self,
                                                             client):
        vehicle = {"preset": "ucits_etf",
                   "settings": {"pricing": {"type": "SwingPricing", "params": {}},
                                "market": {"noise_bps": 9}}}
        settings = validate(client, vehicle=vehicle)["settings"]["vehicle"]["settings"]

        assert settings["pricing"] == {"type": "SwingPricing",
                                       "params": {"factor_bps": None, "threshold": 0.0}}
        assert settings["market"]["noise_bps"] == 9
        assert settings["market"]["persistence"] == 0.5

    def test_findings_point_at_the_field(self,
                                         client):
        validation = validate(client, vehicle={"preset": "uk_oeic", "settings": {
            "pricing": {"type": "SwingPricing", "params": {"threshold": -1}},
            "creation_unit": 10, "launch_price": None}},
            flows=[{"type": "Tides", "params": {}}],
            strategy={"type": "active"})

        assert validation["valid"] is False
        assert paths(validation) == {
            "vehicle.settings.creation_unit": "not_applicable",
            "vehicle.settings.launch_price": "not_nullable",
            "flows.0.type": "unknown_type",
            "strategy.signal": "missing_parameter",
        }

    def test_a_bad_parameter_is_found_inside_its_object(self,
                                                        client):
        validation = validate(client, vehicle={"preset": "uk_oeic", "settings": {
            "pricing": {"type": "SwingPricing", "params": {"threshold": -1, "speed": 2}}}})

        assert paths(validation) == {
            "vehicle.settings.pricing.params.speed": "unknown_parameter"}

        validation = validate(client, vehicle={"preset": "uk_oeic", "settings": {
            "pricing": {"type": "SwingPricing", "params": {"threshold": -1}}}})

        assert paths(validation) == {"vehicle.settings.pricing.params": "invalid_value"}

    def test_an_unknown_preset(self,
                               client):
        assert paths(validate(client, vehicle={"preset": "cayman"})) == {
            "vehicle.preset": "unknown_preset"}

    def test_the_resolved_assumptions_are_complete(self,
                                                   client):
        settings = validate(client, modelling_assumptions={"cash_rate": 0.02})["settings"]

        assert settings["modelling_assumptions"]["cash_rate"] == 0.02
        assert settings["modelling_assumptions"]["periods_per_year"] == 252

    def test_a_bad_request_is_refused_before_it_runs(self,
                                                     client):
        response = client.post("/beacon/BT/backtest", headers=auth(),
                               json={"vehicle": {"preset": "cayman"}})

        assert response.status_code == 422
        assert response.json()["error"]["detail"]["findings"][0]["path"] == "vehicle.preset"


class TestRunning:

    def test_a_plain_request_is_unchanged(self,
                                          client):
        result = run_backtest(client)

        assert result["flows"] == [] and result["nav_per_unit"] is None
        assert result["settings"]["strategy"] == {"type": "index"}
        assert result["settings"]["vehicle"] is None

    def test_a_vehicle_with_flows(self,
                                  client):
        result = run_backtest(client, vehicle=UK_OEIC, transaction_cost_bps=10,
                              flows=[{"type": "PeriodicFlows",
                                      "params": {"fraction": 0.05}}])

        assert result["flows"] and result["fees_paid"] > 0.0
        assert result["metrics"]["money_weighted_return"] is not None
        assert result["nav_per_unit"]["index"] == result["level"]["index"]
        assert result["flows"][0]["adjustment"] > 0.0

    def test_an_etf_is_quoted(self,
                              client):
        result = run_backtest(client, vehicle={"preset": "us_etf",
                                               "settings": {"creation_unit": 10}})

        assert result["market"]["premium"]["index"] == result["level"]["index"]
        assert result["metrics"]["average_spread"] > 0.0

    def test_a_replication(self,
                           client):
        result = run_backtest(client, strategy={
            "type": "index_tracking",
            "replication": {"type": "SampledReplication", "params": {"holdings": 1}}})

        assert all(step["holdings"] == 1 for step in result["replication"])

    def test_an_active_strategy(self,
                                client):
        result = run_backtest(client, strategy={
            "type": "active",
            "signal": {"type": "FieldSignal", "params": {
                "field": {"node": "field", "namespace": "market", "name": "close"}}},
            "construction": {"type": "MeanVariance", "params": {"risk_aversion": 5}},
            "lookback_days": 40, "minimum_observations": 20})

        assert result["active"]
        assert any(step["tracking_error"] > 0.0 for step in result["active"])
        assert result["metrics"]["information_ratio"] is not None
        assert result["settings"]["strategy"]["construction"]["type"] == "MeanVariance"

    def test_the_implementation_reaches_the_run(self,
                                                client):
        result = run_backtest(client, implementation={
            "caps": [{"type": "WeightCap", "params": {"max_weight": 0.4}}],
            "redistribution": "cash"})

        assert result["rebalance_steps"][0]["capped"]
        assert result["rebalance_steps"][0]["cash_weight"] > 0.0

    def test_the_saved_run_lists_what_ran(self,
                                          client):
        run_backtest(client, vehicle=UK_OEIC)
        rows = client.get("/beacon/backtests", headers=auth()).json()["backtests"]
        row = next(row for row in rows if row["index_id"] == "BT")

        assert (row["strategy"], row["vehicle"], row["vehicle_preset"], row["currency"]) == (
            "index", "UK OEIC", "uk_oeic", "USD")
