# tests/test_server_backtest_settings.py
"""BN-272: the backtest API's strategies, implementation, flows, vehicles and
modelling assumptions.

Reuses test_server_backtest's two-name index and data. Runs go through the
job, as a client's would; the settings are checked through
`/backtest/validate`, which builds them without running. Since BN-313 the
strategy and implementation live on a stored strategy, so they are checked
where it is saved.
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
        assert entered.put("/strategies/BT",
                           json={"id": "BT", "name": "BT", "index_id": "BT"},
                           headers=auth()).status_code == 200
        yield entered


def validate(client,
             **body) -> dict:
    response = client.post("/strategies/BT/backtest/validate", json=body, headers=auth())
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

    def test_the_run_s_own_settings_are_described(self,
                                                  options):
        """BN-321: the dates, capital and cost a form needs, with their units."""
        fields = {field["name"]: field for field in options["run_settings"]}

        assert (fields["start"]["type"], fields["end"]["type"]) == ("date", "date")
        assert (fields["initial_capital"]["unit"], fields["initial_capital"]["default"]) == (
            "money", 1_000_000.0)
        assert fields["transaction_cost_bps"]["unit"] == "bps"
        assert fields["dividends"]["choices"] == ["reinvest", "cash", "distribute"]
        assert fields["benchmark.kind"]["choices"] == ["index", "identifier"]

    def test_every_field_of_a_backtest_request_is_described(self,
                                                            client,
                                                            options):
        """A form built from the options can set everything the request takes;
        the analysis is described by its own catalogue (BN-400)."""
        from beacon.server.schemas import StrategyBacktestRequest

        analysis = client.get("/analysis/options", headers=auth()).json()
        run = {field["name"].split(".")[0] for field in options["run_settings"]}
        described = {*run,
                     "flows" if options["flows"] else None,
                     "vehicle" if options["presets"] and options["vehicle_settings"] else None,
                     "share_classes" if options["share_class_settings"] else None,
                     "modelling_assumptions" if options["modelling_assumptions"] else None,
                     "analysis" if analysis["settings"] and analysis["reports"] else None}

        assert set(StrategyBacktestRequest.model_fields) <= described

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
            flows=[{"type": "Tides", "params": {}}])

        assert validation["valid"] is False
        assert paths(validation) == {
            "vehicle.settings.creation_unit": "not_applicable",
            "vehicle.settings.launch_price": "not_nullable",
            "flows.0.type": "unknown_type",
        }

    def test_a_strategy_s_findings_point_at_the_field_where_it_is_saved(self,
                                                                       client):
        """The same paths a backtest request gave them before BN-313."""
        validation = client.post("/strategies/validate",
                                 json={"id": "checked", "name": "Checked", "index_id": "BT",
                                       "strategy": {"type": "active"}},
                                 headers=auth()).json()

        assert validation["valid"] is False
        assert paths(validation) == {"strategy.signal": "missing_parameter"}

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
        response = client.post("/strategies/BT/backtest", headers=auth(),
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

    def test_a_construction_without_a_signal(self,
                                             client):
        result = run_backtest(client, strategy={
            "type": "active", "construction": {"type": "MinimumVariance", "params": {}},
            "lookback_days": 40, "minimum_observations": 20})

        # The first rebalances may have too little history to solve, and hold
        # the benchmark, as the other active tests allow for.
        assert any(step["risk_contributions"] for step in result["active"])

    def test_factor_tilts_take_signals_as_factors(self,
                                                  client):
        factor = {"type": "FieldSignal", "params": {
            "field": {"node": "field", "namespace": "market", "name": "close"}}}
        result = run_backtest(client, strategy={
            "type": "active",
            "construction": {"type": "FactorTilts", "params": {
                "factors": {"price": factor}, "targets": {"price": 0.2}}},
            "lookback_days": 40, "minimum_observations": 20})

        assert any("price" in step["exposures"] for step in result["active"])
        assert result["settings"]["strategy"]["construction"]["params"]["factors"] == {
            "price": {"type": "FieldSignal", "params": {
                "field": factor["params"]["field"], "higher_is_better": True}}}

    def test_a_scored_construction_still_needs_a_signal(self,
                                                        client):
        """Refused where the strategy is saved, before any backtest of it."""
        response = client.put("/strategies/scored", json={
            "id": "scored", "name": "Scored", "index_id": "BT",
            "strategy": {"type": "active", "construction": {"type": "TopN",
                                                            "params": {"n": 2}}}},
            headers=auth())

        assert response.status_code == 422
        assert "signal" in response.text

    def test_the_implementation_reaches_the_run(self,
                                                client):
        result = run_backtest(client, implementation={
            "caps": [{"type": "WeightCap", "params": {"max_weight": 0.4}}],
            "redistribution": "cash"})

        assert result["rebalance_steps"][0]["capped"]
        assert result["rebalance_steps"][0]["cash_weight"] > 0.0

    def test_futures_can_equitise_the_cash_buffer(self,
                                                  client):
        result = run_backtest(client, implementation={
            "cash_buffer": 0.05, "equitise": {"roll_days": 3, "roll_cost_bps": 1.0}})

        assert result["settings"]["implementation"]["equitise"] == {
            "roll_days": 3, "roll_cost_bps": 1.0}

    def test_a_futures_based_replication(self,
                                         client):
        result = run_backtest(client, strategy={
            "type": "index_tracking",
            "replication": {"type": "FuturesReplication", "params": {"roll_cost_bps": 2.0}}})

        assert all(step["holdings"] == 0 for step in result["replication"])

    def test_the_saved_run_lists_what_ran(self,
                                          client):
        run_backtest(client, vehicle=UK_OEIC)
        rows = client.get("/strategies/runs", headers=auth()).json()["backtests"]
        row = next(row for row in rows if row["strategy_id"] == "BT")

        assert (row["strategy"], row["vehicle"], row["vehicle_preset"], row["currency"]) == (
            "index", "UK OEIC", "uk_oeic", "USD")


class TestReinvestInto:
    """BN-343: where reinvested dividends go is a run setting."""

    def test_is_described_with_payer_as_the_default(self,
                                                    options):
        setting = next(spec for spec in options["run_settings"] if spec["name"] == "reinvest_into")

        assert setting["default"] == "payer"
        assert setting["choices"] == ["payer", "holdings"]
        assert options["choices"]["reinvest_into"] == ["payer", "holdings"]

    def test_an_unknown_target_is_a_finding(self,
                                            client):
        assert paths(validate(client, reinvest_into="everything")) == {
            "reinvest_into": "invalid_value"}


class TestTheCashBand:
    """BN-358: the cash band and the futures' whole contracts, multiplier and
    band are request settings."""

    def test_they_reach_the_run(self,
                                client):
        result = run_backtest(client, implementation={
            "cash_buffer": 0.05, "cash_band": {"lower": 0.03, "upper": 0.07},
            "equitise": {"whole_contracts": False, "multiplier": 10.0, "band": 0.01}})
        implementation = result["settings"]["implementation"]

        assert implementation["cash_band"] == {"lower": 0.03, "upper": 0.07}
        assert implementation["equitise"] == {"whole_contracts": False, "multiplier": 10.0,
                                              "band": 0.01}

    def test_futures_pricing_and_costs_reach_the_run(self,
                                                     client):
        # BN-409.
        equitise = {"cost_per_contract": 2.0, "trading_cost_bps": 1.0, "mispricing_bps": 5.0,
                    "mispricing_half_life": 20.0, "mispricing_seed": 7}
        result = run_backtest(client, implementation={"cash_buffer": 0.05,
                                                      "equitise": equitise})

        assert result["settings"]["implementation"]["equitise"] == equitise

    @staticmethod
    def saved(client,
              implementation: dict):
        """Validate a stored strategy carrying *implementation*."""
        return client.post("/strategies/validate", headers=auth(),
                           json={"id": "S", "name": "S", "index_id": "BT",
                                 "implementation": implementation})

    def test_a_band_that_misses_the_buffer_is_found(self,
                                                     client):
        response = self.saved(client, {"cash_buffer": 0.05,
                                       "cash_band": {"lower": 0.06, "upper": 0.07}})

        assert "cash_band" in response.text

    @pytest.mark.parametrize("equitise", [{"whole_contracts": 1}, {"band": True},
                                          {"underlying": 3}, {"mispricing_bps": "wide"}])
    def test_a_flag_and_a_number_are_not_mixed_up(self,
                                                  client,
                                                  equitise):
        response = self.saved(client, {"equitise": equitise})

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_the_options_describe_them(self,
                                       options):
        described = {field["name"] for field in options["implementation_settings"]}

        assert {"cash_band.lower", "cash_band.upper", "equitise.multiplier",
                "equitise.whole_contracts", "equitise.band"} <= described
