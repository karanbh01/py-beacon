# tests/test_server_derivatives.py
"""BN-74: stateless derivatives pricing endpoints."""
import math
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from beacon.testing import dataset

TOKEN = "test-token-value"

SPOT = 100.0
RATE = 0.05
DIVIDEND_YIELD = 0.02
TENOR = 0.5

NOTIONAL = 10_000_000.0


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def futures_body(**overrides) -> dict:
    """A plain half-year contract."""
    body = {"spot": SPOT, "risk_free_rate": RATE,
            "dividend_yield": DIVIDEND_YIELD, "time_to_expiry": TENOR}
    body.update(overrides)

    return body


def trs_body(**overrides) -> dict:
    """A one-year unfunded swap, valued a quarter in."""
    body = {"trade_id": "TRS1",
            "start_date": "2024-01-01",
            "end_date": "2025-01-01",
            "notional": NOTIONAL,
            "spread_bps": 50.0,
            "reference_rate_value": 0.04,
            "payment_frequency": "QUARTERLY",
            "valuation_date": "2024-04-01",
            "last_reset_date": "2024-01-01",
            "spot": 105.0,
            "initial_price": 100.0}
    body.update(overrides)

    return body


@pytest.fixture(scope="module")
def storage():
    with tempfile.TemporaryDirectory() as path:
        yield Path(path)


@pytest.fixture(scope="module")
def client(storage):
    config = ServerConfig(auth_token=TOKEN,
                          data_fetcher=dataset.data_fetcher(),
                          storage_root=storage)

    with TestClient(create_app(config), raise_server_exceptions=False) as started:
        yield started


class TestFuturesFairValue:
    """The acceptance criterion: fair values equal S·e^((r−q)T) to the cent."""

    def test_it_matches_the_closed_form(self,
                                        client):
        payload = client.post("/derivatives/futures/price",
                              json=futures_body(), headers=auth()).json()
        expected = SPOT * math.exp((RATE - DIVIDEND_YIELD) * TENOR)

        assert payload["fair_value"] == pytest.approx(expected, abs=0.005)

    def test_it_matches_across_a_range_of_tenors(self,
                                                 client):
        for tenor in (0.0, 0.25, 1.0, 2.0, 5.0):
            payload = client.post("/derivatives/futures/price",
                                  json=futures_body(time_to_expiry=tenor),
                                  headers=auth()).json()
            expected = SPOT * math.exp((RATE - DIVIDEND_YIELD) * tenor)

            assert payload["fair_value"] == pytest.approx(expected, abs=0.005), tenor

    def test_a_zero_tenor_prices_at_spot(self,
                                         client):
        payload = client.post("/derivatives/futures/price",
                              json=futures_body(time_to_expiry=0.0),
                              headers=auth()).json()

        assert payload["fair_value"] == pytest.approx(SPOT, abs=1e-12)

    def test_borrow_cost_raises_the_forward(self,
                                            client):
        plain = client.post("/derivatives/futures/price", json=futures_body(),
                            headers=auth()).json()
        borrowed = client.post("/derivatives/futures/price",
                               json=futures_body(borrow_cost=0.01),
                               headers=auth()).json()

        assert borrowed["fair_value"] > plain["fair_value"]

    def test_dates_are_used_when_both_are_given(self,
                                                client):
        """Less ambiguous than a hand-typed tenor beside an expiry."""
        payload = client.post("/derivatives/futures/price",
                              json=futures_body(valuation_date="2024-01-01",
                                                expiry="2024-07-01",
                                                time_to_expiry=99.0),
                              headers=auth()).json()

        assert payload["time_to_expiry"] == pytest.approx(182 / 365.0, abs=1e-9)

    def test_an_expiry_before_valuation_is_refused(self,
                                                   client):
        response = client.post("/derivatives/futures/price",
                               json=futures_body(valuation_date="2024-07-01",
                                                 expiry="2024-01-01"),
                               headers=auth())

        assert response.status_code == 404

    def test_no_tenor_at_all_is_refused(self,
                                        client):
        response = client.post("/derivatives/futures/price",
                               json={"spot": SPOT}, headers=auth())

        assert response.status_code == 404

    def test_a_negative_tenor_is_refused(self,
                                         client):
        response = client.post("/derivatives/futures/price",
                               json=futures_body(time_to_expiry=-1.0),
                               headers=auth())

        assert response.status_code == 404


class TestCurve:

    def test_a_flat_curve_reproduces_the_scalar_rate(self,
                                                     client):
        """The property BN-96 established, holding at the API boundary."""
        scalar = client.post("/derivatives/futures/price", json=futures_body(),
                             headers=auth()).json()
        curved = client.post("/derivatives/futures/price",
                             json=futures_body(curve={"1.0": RATE}),
                             headers=auth()).json()

        assert curved["fair_value"] == scalar["fair_value"]

    def test_a_shaped_curve_prices_off_the_right_pillar(self,
                                                        client):
        payload = client.post("/derivatives/futures/price",
                              json=futures_body(time_to_expiry=1.0,
                                                curve={"0.25": 0.03, "1.0": 0.06}),
                              headers=auth()).json()

        assert payload["financing_rate"] == pytest.approx(0.06, abs=1e-12)

    def test_the_curve_is_interpolated_between_pillars(self,
                                                       client):
        payload = client.post("/derivatives/futures/price",
                              json=futures_body(time_to_expiry=0.625,
                                                curve={"0.25": 0.03, "1.0": 0.04}),
                              headers=auth()).json()

        assert payload["financing_rate"] == pytest.approx(0.035, abs=1e-12)


class TestCarryDecomposition:

    @pytest.fixture(scope="class")
    def carry(self,
              client):
        return client.post("/derivatives/futures/price",
                           json=futures_body(borrow_cost=0.005),
                           headers=auth()).json()["carry"]

    def test_the_total_is_fair_value_minus_spot(self,
                                                client,
                                                carry):
        payload = client.post("/derivatives/futures/price",
                              json=futures_body(borrow_cost=0.005),
                              headers=auth()).json()

        assert carry["total"] == pytest.approx(payload["fair_value"] - SPOT,
                                               abs=1e-12)

    def test_financing_adds_and_dividends_subtract(self,
                                                   carry):
        assert carry["financing"] > 0.0
        assert carry["dividend"] < 0.0
        assert carry["borrow"] > 0.0

    def test_the_residual_accounts_for_the_rest(self,
                                                carry):
        """Carry compounds rather than adds, so the parts do not sum exactly.
        The gap is reported rather than spread across them, which would make
        each slightly wrong to hide that the split is approximate."""
        parts = carry["financing"] + carry["dividend"] + carry["borrow"]

        assert carry["total"] == pytest.approx(parts + carry["residual"], abs=1e-12)

    def test_the_residual_is_small_relative_to_the_carry(self,
                                                         carry):
        assert abs(carry["residual"]) < abs(carry["total"])


class TestFuturesExtras:

    def test_contract_value_scales_with_multiplier_and_count(self,
                                                             client):
        payload = client.post("/derivatives/futures/price",
                              json=futures_body(contract_multiplier=50.0,
                                                contracts=3.0),
                              headers=auth()).json()

        assert payload["contract_value"] == pytest.approx(
            payload["fair_value"] * 150.0, abs=1e-9)

    def test_basis_and_implied_repo_need_a_quote(self,
                                                 client):
        """Computing them against the theoretical value would make both
        identically zero, which says nothing."""
        payload = client.post("/derivatives/futures/price", json=futures_body(),
                              headers=auth()).json()

        assert payload["basis"] is None
        assert payload["implied_repo"] is None

    def test_a_rich_quote_gives_a_positive_basis_and_a_higher_repo(self,
                                                                   client):
        payload = client.post("/derivatives/futures/price",
                              json=futures_body(market_price=103.0),
                              headers=auth()).json()

        assert payload["basis"] > 0.0
        assert payload["implied_repo"] > RATE

    def test_discrete_dividends_lower_the_forward(self,
                                                  client):
        plain = client.post("/derivatives/futures/price",
                            json=futures_body(dividend_yield=0.0),
                            headers=auth()).json()
        discrete = client.post("/derivatives/futures/price",
                               json=futures_body(dividend_yield=0.0,
                                                 dividends=[[0.25, 2.0]]),
                               headers=auth()).json()

        assert discrete["fair_value"] < plain["fair_value"]

    def test_the_sensitivity_grid_is_rectangular(self,
                                                 client):
        grid = client.post("/derivatives/futures/price", json=futures_body(),
                           headers=auth()).json()["sensitivity"]

        assert len(grid["data"]) == len(grid["index"])
        assert all(len(row) == len(grid["columns"]) for row in grid["data"])

    def test_the_grid_rises_with_the_rate(self,
                                          client):
        grid = client.post("/derivatives/futures/price", json=futures_body(),
                           headers=auth()).json()["sensitivity"]

        for row in grid["data"]:
            assert row == sorted(row)

    def test_grid_axes_can_be_supplied(self,
                                       client):
        grid = client.post("/derivatives/futures/price",
                           json=futures_body(grid_tenors=[0.5, 1.0],
                                             grid_rates=[0.01, 0.02, 0.03]),
                           headers=auth()).json()["sensitivity"]

        assert len(grid["index"]) == 2
        assert len(grid["columns"]) == 3


class TestTrsAccruals:
    """The acceptance criterion: accruals are exact ACT/360."""

    @pytest.fixture(scope="class")
    def payload(self,
                client):
        return client.post("/derivatives/trs/price", json=trs_body(),
                           headers=auth()).json()

    def test_the_accrual_fraction_is_act_360(self,
                                             payload):
        """91 days from 1 January to 1 April in a leap year."""
        assert payload["accrual_days"] == 91
        assert payload["accrual_fraction"] == pytest.approx(91 / 360.0, rel=1e-15)

    def test_the_financing_leg_is_exact(self,
                                        payload):
        """notional x (reference + spread) x days/360."""
        expected = NOTIONAL * (0.04 + 0.005) * (91 / 360.0)

        assert payload["financing_leg"] == pytest.approx(expected, rel=1e-12)

    def test_the_total_return_leg_is_exact(self,
                                           payload):
        assert payload["total_return_leg"] == pytest.approx(
            NOTIONAL * (105.0 / 100.0 - 1.0), rel=1e-12)

    def test_the_present_value_is_the_difference(self,
                                                 payload):
        assert payload["present_value"] == pytest.approx(
            payload["total_return_leg"] - payload["financing_leg"], rel=1e-12)

    def test_every_schedule_period_uses_act_360(self,
                                                payload):
        for period in payload["schedule"]:
            assert period["accrual_fraction"] == pytest.approx(
                period["days"] / 360.0, rel=1e-15)

    def test_each_period_amount_is_exact(self,
                                         payload):
        for period in payload["schedule"]:
            expected = NOTIONAL * (period["rate"] + 0.005) * period["accrual_fraction"]
            assert period["amount"] == pytest.approx(expected, rel=1e-12)

    def test_the_schedule_covers_the_remaining_life(self,
                                                    payload):
        assert payload["schedule"][0]["start"] == "2024-01-01"
        assert payload["schedule"][-1]["end"] == "2025-01-01"

    def test_the_periods_are_contiguous(self,
                                        payload):
        for earlier, later in zip(payload["schedule"], payload["schedule"][1:],
                                  strict=False):
            assert earlier["end"] == later["start"]


class TestTrsDv01:

    def test_it_matches_the_hand_calculation(self,
                                             client):
        """notional x 1bp x 91/360, negative to the receiver."""
        payload = client.post("/derivatives/trs/price", json=trs_body(),
                              headers=auth()).json()
        expected = -NOTIONAL * 0.0001 * (91 / 360.0)

        assert payload["dv01"] == pytest.approx(expected, rel=1e-12)

    def test_a_funded_swap_has_no_rate_sensitivity(self,
                                                   client):
        """Only the spread accrues, and it does not move with the rate."""
        payload = client.post("/derivatives/trs/price",
                              json=trs_body(reset_type="FUNDED"),
                              headers=auth()).json()

        assert payload["dv01"] == 0.0

    def test_the_sign_is_negative_for_a_receiver(self,
                                                 client):
        payload = client.post("/derivatives/trs/price", json=trs_body(),
                              headers=auth()).json()

        assert payload["dv01"] < 0.0


class TestTrsCurveProjection:

    def test_the_current_period_uses_the_fixed_rate(self,
                                                    client):
        """Projecting it off a curve would overwrite an observed number with a
        modelled one."""
        payload = client.post("/derivatives/trs/price",
                              json=trs_body(curve={"0.25": 0.08, "2.0": 0.09}),
                              headers=auth()).json()

        assert payload["schedule"][0]["rate"] == pytest.approx(0.04, abs=1e-12)

    def test_later_periods_are_projected_forward(self,
                                                 client):
        payload = client.post("/derivatives/trs/price",
                              json=trs_body(curve={"0.25": 0.08, "2.0": 0.09}),
                              headers=auth()).json()

        assert payload["schedule"][-1]["rate"] > 0.05

    def test_a_flat_curve_leaves_every_period_at_the_rate(self,
                                                          client):
        payload = client.post("/derivatives/trs/price",
                              json=trs_body(curve={"1.0": 0.04}),
                              headers=auth()).json()

        for period in payload["schedule"]:
            assert period["rate"] == pytest.approx(0.04, abs=1e-12)


class TestTrsExtras:

    def test_the_fair_spread_zeroes_the_trade(self,
                                              client):
        """Repricing at the fair spread should leave nothing on the table."""
        payload = client.post("/derivatives/trs/price", json=trs_body(),
                              headers=auth()).json()

        repriced = client.post(
            "/derivatives/trs/price",
            json=trs_body(spread_bps=payload["fair_spread_bps"]),
            headers=auth()).json()

        assert repriced["present_value"] == pytest.approx(0.0, abs=1e-6)

    def test_no_accrual_means_no_fair_spread(self,
                                             client):
        """With a zero day count no spread could balance the trade."""
        payload = client.post("/derivatives/trs/price",
                              json=trs_body(valuation_date="2024-01-01"),
                              headers=auth()).json()

        assert payload["fair_spread_bps"] is None

    def test_the_breakeven_table_needs_prices_and_a_tenor(self,
                                                          client):
        payload = client.post("/derivatives/trs/price", json=trs_body(),
                              headers=auth()).json()

        assert payload["breakeven"] == []

    def test_a_richer_future_implies_a_wider_breakeven_spread(self,
                                                              client):
        payload = client.post("/derivatives/trs/price",
                              json=trs_body(time_to_expiry=0.75,
                                            dividend_yield=0.02,
                                            futures_prices=[100.0, 103.0, 106.0]),
                              headers=auth()).json()
        spreads = [row["breakeven_spread_bps"] for row in payload["breakeven"]]

        assert spreads == sorted(spreads)


class TestTermStructureAndRoll:

    def test_the_strip_prices_every_expiry(self,
                                           client):
        payload = client.get("/derivatives/AAA/term-structure",
                             params={"expiries": ["2026-03-20", "2026-06-19",
                                                  "2026-09-18"],
                                     "risk_free_rate": 0.05,
                                     "dividend_yield": 0.02},
                             headers=auth()).json()

        assert len(payload["entries"]) == 3
        assert payload["spot"] > 0.0

    def test_positive_carry_makes_later_contracts_dearer(self,
                                                         client):
        payload = client.get("/derivatives/AAA/term-structure",
                             params={"expiries": ["2026-03-20", "2026-09-18"],
                                     "risk_free_rate": 0.05},
                             headers=auth()).json()
        prices = [entry["theoretical"] for entry in payload["entries"]]

        assert prices == sorted(prices)

    def test_an_unknown_index_is_a_404(self,
                                       client):
        response = client.get("/derivatives/NOPE/term-structure",
                              params={"expiries": ["2026-03-20"]}, headers=auth())

        assert response.status_code == 404

    def test_contango_gives_a_negative_roll(self,
                                            client):
        """Positive carry means the back contract is dearer, so rolling costs."""
        payload = client.get("/derivatives/AAA/roll",
                             params={"front_expiry": "2026-03-20",
                                     "back_expiry": "2026-06-19",
                                     "risk_free_rate": 0.05},
                             headers=auth()).json()

        assert payload["roll_cost"] > 0.0
        assert payload["annualised_roll"] < 0.0

    def test_backwardation_gives_a_positive_roll(self,
                                                 client):
        """A dividend yield above the financing rate inverts the curve."""
        payload = client.get("/derivatives/AAA/roll",
                             params={"front_expiry": "2026-03-20",
                                     "back_expiry": "2026-06-19",
                                     "risk_free_rate": 0.01,
                                     "dividend_yield": 0.06},
                             headers=auth()).json()

        assert payload["roll_cost"] < 0.0
        assert payload["annualised_roll"] > 0.0

    def test_a_back_expiry_before_the_front_is_refused(self,
                                                       client):
        response = client.get("/derivatives/AAA/roll",
                              params={"front_expiry": "2026-06-19",
                                      "back_expiry": "2026-03-20"},
                              headers=auth())

        assert response.status_code == 404

    def test_it_requires_authentication(self,
                                        client):
        assert client.get("/derivatives/AAA/roll",
                          params={"front_expiry": "2026-03-20",
                                  "back_expiry": "2026-06-19"}).status_code == 401


class TestStatelessness:
    """The acceptance criterion: no state written anywhere."""

    def _tree(self,
              root: Path) -> set[str]:
        return {str(path.relative_to(root)) for path in root.rglob("*")}

    def test_pricing_writes_nothing(self,
                                    client,
                                    storage):
        before = self._tree(storage)

        client.post("/derivatives/futures/price", json=futures_body(),
                    headers=auth())
        client.post("/derivatives/trs/price", json=trs_body(), headers=auth())

        assert self._tree(storage) == before

    def test_reads_write_nothing(self,
                                 client,
                                 storage):
        before = self._tree(storage)

        client.get("/derivatives/AAA/term-structure",
                   params={"expiries": ["2026-03-20"]}, headers=auth())
        client.get("/derivatives/AAA/roll",
                   params={"front_expiry": "2026-03-20",
                           "back_expiry": "2026-06-19"}, headers=auth())

        assert self._tree(storage) == before

    def test_pricing_creates_no_jobs(self,
                                     client):
        """Stateless also means it does not queue work behind the caller's
        back."""
        before = len(client.get("/jobs", headers=auth()).json()["jobs"])

        client.post("/derivatives/futures/price", json=futures_body(),
                    headers=auth())

        assert len(client.get("/jobs", headers=auth()).json()["jobs"]) == before

    def test_the_same_request_gives_the_same_answer(self,
                                                    client):
        """A pure function of its input, so repetition is free and identical."""
        first = client.post("/derivatives/futures/price", json=futures_body(),
                            headers=auth()).json()
        second = client.post("/derivatives/futures/price", json=futures_body(),
                             headers=auth()).json()

        assert first == second


class TestWithoutADataSource:

    def test_pricing_still_works(self):
        """It needs no data at all, which is the point of being stateless."""
        with tempfile.TemporaryDirectory() as path:
            config = ServerConfig(auth_token=TOKEN, storage_root=Path(path))

            with TestClient(create_app(config),
                            raise_server_exceptions=False) as bare:
                response = bare.post("/derivatives/futures/price",
                                     json=futures_body(), headers=auth())

        assert response.status_code == 200

    def test_index_reads_report_the_missing_source(self):
        with tempfile.TemporaryDirectory() as path:
            config = ServerConfig(auth_token=TOKEN, storage_root=Path(path))

            with TestClient(create_app(config),
                            raise_server_exceptions=False) as bare:
                response = bare.get("/derivatives/AAA/term-structure",
                                    params={"expiries": ["2026-03-20"]},
                                    headers=auth())

        assert response.status_code == 500


class TestOpenApi:

    def test_every_endpoint_is_documented(self,
                                          client):
        paths = client.app.openapi()["paths"]

        for path in ("/derivatives/futures/price",
                     "/derivatives/trs/price",
                     "/derivatives/{index_id}/term-structure",
                     "/derivatives/{index_id}/roll"):
            assert path in paths, path


class TestDegenerateInputs:

    def test_a_swap_valued_at_maturity_has_an_empty_schedule(self,
                                                             client):
        """Nothing left to accrue, so the schedule is empty rather than
        carrying a zero-length period."""
        payload = client.post("/derivatives/trs/price",
                              json=trs_body(last_reset_date="2025-01-01",
                                            valuation_date="2025-01-01"),
                              headers=auth()).json()

        assert payload["schedule"] == []
        assert payload["accrual_days"] == 0

    def test_an_identifier_with_no_usable_price_is_a_404(self):
        """The FX pair carries prices, but a name whose closes are all missing
        cannot be priced and should say so rather than serve a nan."""
        import pandas as pd

        from beacon.data.base import MarketData
        from beacon.data.fetcher import DataFetcher

        blank = MarketData.from_dataframe(pd.DataFrame({
            "IDENTIFIER": ["BLANK"] * 3,
            "DATE": pd.bdate_range("2024-01-01", periods=3),
            "CLOSE": [float("nan")] * 3}))

        with tempfile.TemporaryDirectory() as path:
            config = ServerConfig(auth_token=TOKEN,
                                  data_fetcher=DataFetcher(blank),
                                  storage_root=Path(path))

            with TestClient(create_app(config),
                            raise_server_exceptions=False) as client:
                response = client.get("/derivatives/BLANK/term-structure",
                                      params={"expiries": ["2026-03-20"]},
                                      headers=auth())

        assert response.status_code == 404
