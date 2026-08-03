# tests/test_risk_contribution.py
"""BN-123: which holdings drive an index's risk.

A weights table says what the index owns; it does not say what the index is
exposed to. The two differ enough to matter, and a table without this column
looks like a risk view without being one.

The centrepiece is the summation identity. Contributions sum to the portfolio
volatility **exactly** — it follows from Euler's theorem, not from an
approximation — so the test has no tolerance to hide behind. Everything else
here is about the cases where the honest answer is "not all of it".
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.risk.contribution import (
    RiskContributions,
    active_risk_contributions,
    active_weights,
    risk_contributions,
)
from beacon.server import ServerConfig, create_app
from beacon.testing import dataset

TOKEN = "test-token-value"
INDEX_ID = "risky"
START = "2023-01-02"
END = "2024-06-28"

# Two assets: 20% and 10% annualised volatility, correlated 0.5.
COVARIANCE = pd.DataFrame([[0.04, 0.01], [0.01, 0.01]],
                          index=["AAA", "BBB"], columns=["AAA", "BBB"])


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


class TestTheIdentity:
    """Contributions sum to the whole, exactly."""

    def test_they_sum_to_the_volatility(self):
        result = risk_contributions({"AAA": 0.5, "BBB": 0.5}, COVARIANCE)

        assert sum(result.contribution.values()) == pytest.approx(
            result.volatility, abs=1e-15)

    def test_the_volatility_is_the_hand_computed_one(self):
        """w'Sw = 0.25(0.04) + 2(0.25)(0.01) + 0.25(0.01) = 0.0175."""
        result = risk_contributions({"AAA": 0.5, "BBB": 0.5}, COVARIANCE)

        assert result.volatility == pytest.approx(0.0175 ** 0.5, abs=1e-12)

    @pytest.mark.parametrize("weights", [
        {"AAA": 1.0, "BBB": 0.0},
        {"AAA": 0.9, "BBB": 0.1},
        {"AAA": 0.25, "BBB": 0.75},
        {"AAA": 0.0, "BBB": 1.0},
    ])
    def test_it_holds_at_any_weights(self, weights):
        result = risk_contributions(weights, COVARIANCE)

        assert sum(result.contribution.values()) == pytest.approx(
            result.volatility, abs=1e-15)

    def test_it_holds_on_a_real_estimate(self):
        """A six-name covariance estimated from the canonical dataset, where
        the matrix is full rather than hand-written."""
        from beacon.risk import estimate_risk_model

        model = estimate_risk_model(dataset.returns())
        weights = {name: 1.0 / len(dataset.UNIVERSE)
                   for name in dataset.UNIVERSE}

        result = risk_contributions(weights, model.covariance)

        assert sum(result.contribution.values()) == pytest.approx(
            result.volatility, abs=1e-12)

    def test_a_concentrated_holding_dominates(self):
        """Sanity: the name carrying almost all the weight carries almost all
        the risk. Without this the identity could hold on nonsense."""
        result = risk_contributions({"AAA": 0.99, "BBB": 0.01}, COVARIANCE)

        assert result.contribution["AAA"] > 20 * result.contribution["BBB"]


class TestRiskIsNotWeight:
    """The reason the column exists."""

    def test_a_volatile_name_contributes_more_than_its_weight(self):
        """AAA is twice as volatile as BBB. At equal weights it must account
        for well over half the risk — if contribution tracked weight, the
        column would be telling the reader nothing new."""
        result = risk_contributions({"AAA": 0.5, "BBB": 0.5}, COVARIANCE)
        share = result.contribution["AAA"] / result.volatility

        assert share > 0.6

    def test_a_smaller_holding_can_carry_more_risk(self):
        """Ordering by weight and ordering by risk genuinely disagree.

        Not at every weighting, which is worth knowing: with these two assets
        the crossover sits near 34%, so at 30/70 the larger holding does carry
        more risk and at 40/60 it does not. The column is informative because
        the two orderings differ *somewhere*, not because they always differ.
        """
        result = risk_contributions({"AAA": 0.4, "BBB": 0.6}, COVARIANCE)

        assert result.contribution["AAA"] > result.contribution["BBB"]

    def test_the_orderings_agree_when_the_weight_gap_is_wide_enough(self):
        """The other side of it, so the previous test is not read as "risk
        always inverts weight"."""
        result = risk_contributions({"AAA": 0.3, "BBB": 0.7}, COVARIANCE)

        assert result.contribution["BBB"] > result.contribution["AAA"]


class TestPartialCoverage:
    """Names the model has no estimate for."""

    def test_an_uncovered_name_is_reported_not_dropped_silently(self):
        result = risk_contributions({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
                                    COVARIANCE)

        assert result.uncovered == ("CCC",)
        assert not result.is_complete

    def test_covered_weight_says_how_much_the_figure_speaks_for(self):
        result = risk_contributions({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
                                    COVARIANCE)

        assert result.covered_weight == pytest.approx(0.8)

    def test_the_covered_names_keep_their_real_weights(self):
        """Not renormalised. Renormalising would claim the index holds more of
        the covered names than it does, and restate the portfolio to make a
        number tidier."""
        partial = risk_contributions({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
                                     COVARIANCE)
        as_held = risk_contributions({"AAA": 0.5, "BBB": 0.3}, COVARIANCE)

        assert partial.volatility == pytest.approx(as_held.volatility)

    def test_the_identity_still_holds_over_the_covered_part(self):
        result = risk_contributions({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
                                    COVARIANCE)

        assert sum(result.contribution.values()) == pytest.approx(
            result.volatility, abs=1e-15)

    def test_full_coverage_reports_complete(self):
        result = risk_contributions({"AAA": 0.5, "BBB": 0.5}, COVARIANCE)

        assert result.is_complete
        assert result.covered_weight == pytest.approx(1.0)

    def test_nothing_covered_gives_no_volatility(self):
        result = risk_contributions({"XXX": 1.0}, COVARIANCE)

        assert result.volatility == 0.0
        assert result.uncovered == ("XXX",)


class TestDegenerateInputs:
    """Where a decomposition is not possible."""

    def test_no_weights(self):
        assert risk_contributions({}, COVARIANCE).volatility == 0.0

    def test_no_covariance(self):
        assert risk_contributions({"AAA": 1.0}, pd.DataFrame()).volatility == 0.0

    def test_a_zero_variance_portfolio(self, caplog):
        """Dividing by a volatility of zero would produce infinities on a
        pane; reporting nothing is the honest answer."""
        import logging

        zeros = pd.DataFrame(np.zeros((2, 2)), index=["AAA", "BBB"],
                             columns=["AAA", "BBB"])

        with caplog.at_level(logging.WARNING):
            result = risk_contributions({"AAA": 0.5, "BBB": 0.5}, zeros)

        assert result.volatility == 0.0
        assert result.contribution == {}
        assert "no decomposition" in caplog.text

    def test_a_flat_result_is_still_a_result(self):
        assert isinstance(risk_contributions({}, COVARIANCE), RiskContributions)


class TestServedOnTheWeightsPane:
    """`GET /beacon/{id}/weights?risk=true`."""

    @pytest.fixture(scope="class")
    def client(self):
        with tempfile.TemporaryDirectory() as storage:
            config = ServerConfig(auth_token=TOKEN,
                                  data_fetcher=dataset.data_fetcher(),
                                  storage_root=Path(storage))

            with TestClient(create_app(config),
                            raise_server_exceptions=False) as started:
                started.put(f"/indices/{INDEX_ID}", json={
                    "id": INDEX_ID, "name": "Risky", "base_date": START,
                    "base_value": 1000.0, "currency": "USD",
                    "rebalancing_frequency": "QUARTERLY", "description": None,
                    "universe": {"universe_id": None,
                                 "identifiers": list(dataset.UNIVERSE)},
                    "pipeline": {
                        "selection": [],
                        "weighting": {"id": "w", "scheme": "MarketCapWeighted",
                                      "params": {}, "max_weight": None},
                        "treatment": {"corporate_actions": "ADJUST_DIVISOR"},
                    }}, headers=auth())
                started.post(f"/beacon/{INDEX_ID}/backtest",
                             json={"start": START, "end": END,
                                   "transaction_cost_bps": 10.0},
                             headers=auth())
                started.portal.call(started.app.state.jobs.drain)

                yield started

    def weights(self, client, **params):
        response = client.get(f"/beacon/{INDEX_ID}/weights", params=params,
                              headers=auth())

        assert response.status_code == 200, response.text

        return response.json()

    def test_it_is_absent_by_default(self, client):
        """Estimating a covariance over every constituent is the pane's whole
        cost, and nobody should pay it without asking."""
        body = self.weights(client)

        assert body["risk"] is None
        assert all(row["risk_contribution"] is None for row in body["rows"])

    def test_it_appears_when_requested(self, client):
        body = self.weights(client, risk="true")

        assert body["risk"] is not None
        assert body["risk"]["volatility"] > 0.0

    def test_every_row_carries_a_contribution(self, client):
        body = self.weights(client, risk="true")

        assert all(row["risk_contribution"] is not None for row in body["rows"])

    def test_the_rows_sum_to_the_reported_volatility(self, client):
        """The identity, end to end. If the table and the summary above it can
        disagree, the pane is showing two different indices."""
        body = self.weights(client, risk="true")

        total = sum(row["risk_contribution"] for row in body["rows"])

        assert total == pytest.approx(body["risk"]["volatility"], abs=1e-9)

    def test_the_estimation_window_is_the_run(self, client):
        """A decomposition of this index's risk estimated over some other
        period would describe a different history from the levels beside it."""
        body = self.weights(client, risk="true")

        assert body["risk"]["window_start"] == START
        assert body["risk"]["window_end"] == END

    def test_the_whole_index_is_covered(self, client):
        body = self.weights(client, risk="true")

        assert body["risk"]["covered_weight"] == pytest.approx(1.0, abs=1e-9)
        assert body["risk"]["uncovered"] == []

    def test_the_volatility_is_plausible(self, client):
        """The canonical dataset's names run at a few tens of percent, so an
        equal-ish basket of six should land in the same neighbourhood. A wildly
        different figure would mean the annualisation or the window was
        wrong."""
        body = self.weights(client, risk="true")

        assert 0.02 < body["risk"]["volatility"] < 1.0

    def test_the_other_columns_are_unaffected(self, client):
        """Asking for risk must not change what the pane already reported."""
        plain = self.weights(client)
        with_risk = self.weights(client, risk="true")

        assert plain["weights"] == with_risk["weights"]
        assert ([row["identifier"] for row in plain["rows"]]
                == [row["identifier"] for row in with_risk["rows"]])


class TestActiveRisk:
    """Tracking error, decomposed across active positions."""

    def test_the_identity_holds_on_active_weights(self):
        result = active_risk_contributions({"AAA": 0.7, "BBB": 0.3},
                                           {"AAA": 0.5, "BBB": 0.5},
                                           COVARIANCE)

        assert sum(result.contribution.values()) == pytest.approx(
            result.volatility, abs=1e-15)

    def test_the_tracking_error_is_the_hand_computed_one(self):
        """Active weights are [0.2, -0.2], so the active variance is
        0.04(0.04) - 2(0.04)(0.01) + 0.04(0.01) = 0.0012."""
        result = active_risk_contributions({"AAA": 0.7, "BBB": 0.3},
                                           {"AAA": 0.5, "BBB": 0.5},
                                           COVARIANCE)

        assert result.volatility == pytest.approx(0.0012 ** 0.5, abs=1e-12)

    def test_matching_the_benchmark_is_zero_tracking_error(self):
        weights = {"AAA": 0.5, "BBB": 0.5}

        assert active_risk_contributions(weights, weights,
                                         COVARIANCE).volatility == 0.0

    def test_a_contribution_can_be_negative(self):
        """The property that must not be hidden behind an absolute value.

        A negative contribution needs the position to point *against* the
        book's overall active exposure — it is not enough to be underweight,
        since an underweight in a name the portfolio is also underweight
        overall contributes positively. Here the portfolio holds 20% cash and
        is therefore net underweight, while still being overweight BBB: that
        overweight hedges, and genuinely reduces tracking error.
        """
        result = active_risk_contributions({"AAA": 0.2, "BBB": 0.6},
                                           {"AAA": 0.5, "BBB": 0.5},
                                           COVARIANCE)

        assert result.contribution["BBB"] < 0.0
        assert result.contribution["AAA"] > 0.0
        assert sum(result.contribution.values()) == pytest.approx(
            result.volatility, abs=1e-15)

    def test_a_name_not_held_is_still_an_active_position(self):
        """Usually the largest one there is. Intersecting the two universes
        instead of taking their union would drop it silently."""
        active = active_weights({"AAA": 1.0}, {"AAA": 0.5, "BBB": 0.5})

        assert active == {"AAA": 0.5, "BBB": -0.5}

    def test_a_name_held_and_not_in_the_benchmark_is_an_overweight(self):
        active = active_weights({"AAA": 0.5, "CCC": 0.5}, {"AAA": 1.0})

        assert active == {"AAA": -0.5, "CCC": 0.5}

    def test_coverage_is_a_share_of_gross_active_weight(self):
        """Active weights sum to roughly zero, so a plain sum would say
        nothing about how much of the position is covered."""
        result = active_risk_contributions({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
                                           {"AAA": 0.4, "BBB": 0.4, "CCC": 0.2},
                                           COVARIANCE)

        assert result.uncovered == ("CCC",)
        assert 0.0 < result.covered_weight <= 1.0

    def test_an_empty_benchmark_is_the_portfolio_itself(self):
        """Everything held becomes an overweight, so active risk equals total
        risk — a degenerate case worth behaving sensibly."""
        weights = {"AAA": 0.5, "BBB": 0.5}

        active = active_risk_contributions(weights, {}, COVARIANCE)
        total = risk_contributions(weights, COVARIANCE)

        assert active.volatility == pytest.approx(total.volatility)


class TestActiveRiskOnThePane:
    """`?risk=true&benchmark=<id>`."""

    @pytest.fixture(scope="class")
    def client(self):
        """Two indices over one universe, weighted differently, so they carry
        a real active position against each other."""
        with tempfile.TemporaryDirectory() as storage:
            config = ServerConfig(auth_token=TOKEN,
                                  data_fetcher=dataset.data_fetcher(),
                                  storage_root=Path(storage))

            with TestClient(create_app(config),
                            raise_server_exceptions=False) as started:
                for index_id, scheme in (("active", "MarketCapWeighted"),
                                         ("bench", "EqualWeighted")):
                    started.put(f"/indices/{index_id}", json={
                        "id": index_id, "name": index_id, "base_date": START,
                        "base_value": 1000.0, "currency": "USD",
                        "rebalancing_frequency": "QUARTERLY",
                        "description": None,
                        "universe": {"universe_id": None,
                                     "identifiers": list(dataset.UNIVERSE)},
                        "pipeline": {
                            "selection": [],
                            "weighting": {"id": "w", "scheme": scheme,
                                          "params": {}, "max_weight": None},
                            "treatment": {"corporate_actions": "ADJUST_DIVISOR"},
                        }}, headers=auth())
                    started.post(f"/beacon/{index_id}/backtest",
                                 json={"start": START, "end": END,
                                       "transaction_cost_bps": 10.0},
                                 headers=auth())

                started.portal.call(started.app.state.jobs.drain)

                yield started

    def weights(self, client, **params):
        response = client.get("/beacon/active/weights", params=params,
                              headers=auth())

        assert response.status_code == 200, response.text

        return response.json()

    def test_it_is_absent_without_a_benchmark(self, client):
        body = self.weights(client, risk="true")

        assert body["active_risk"] is None
        assert all(row["active_risk_contribution"] is None
                   for row in body["rows"])

    def test_it_appears_with_one(self, client):
        body = self.weights(client, risk="true", benchmark="bench")

        assert body["active_risk"] is not None
        assert body["active_risk"]["benchmark"] == "bench"
        assert body["active_risk"]["tracking_error"] > 0.0

    def test_the_contributions_sum_to_the_tracking_error(self, client):
        """Including names the index does not hold, which is why they are
        published separately rather than dropped."""
        body = self.weights(client, risk="true", benchmark="bench")

        from_rows = sum(row["active_risk_contribution"] for row in body["rows"]
                        if row["active_risk_contribution"] is not None)
        from_others = sum(body["active_risk"]["contributions_not_held"].values())

        assert (from_rows + from_others) == pytest.approx(
            body["active_risk"]["tracking_error"], abs=1e-9)

    def test_active_weights_are_reported_per_row(self, client):
        body = self.weights(client, risk="true", benchmark="bench")

        assert all(row["active_weight"] is not None for row in body["rows"])

    def test_the_active_weights_net_out(self, client):
        """Both indices hold the same universe and each sums to one, so the
        active position is self-financing."""
        body = self.weights(client, risk="true", benchmark="bench")

        assert sum(row["active_weight"]
                   for row in body["rows"]) == pytest.approx(0.0, abs=1e-9)

    def test_tracking_error_is_below_total_volatility(self, client):
        """Cap- and equal-weighted versions of one universe move together, so
        the difference between them is far less volatile than either."""
        body = self.weights(client, risk="true", benchmark="bench")

        assert (body["active_risk"]["tracking_error"]
                < body["risk"]["volatility"])

    def test_a_benchmark_needs_risk_to_be_requested(self, client):
        """Asking for a comparison without the decomposition it lives in
        should not silently do half of it."""
        body = self.weights(client, benchmark="bench")

        assert body["active_risk"] is None

    def test_an_unknown_benchmark_is_a_404(self, client):
        response = client.get("/beacon/active/weights",
                              params={"risk": "true", "benchmark": "nope"},
                              headers=auth())

        assert response.status_code == 404
