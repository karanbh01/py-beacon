# tests/test_corporate_action_kind.py
"""BN-118: `kind`, pay date and status on a corporate action.

`CorporateAction` was `{ex_date, type, value}`, so a client had to infer
cash-vs-ratio from a hardcoded list of type strings. That works until a type
the list has never seen arrives, at which point the action renders as whichever
the list defaults to — confidently, and wrongly. A ratio shown as a cash amount
is a number on screen that means something else entirely.

The tests below are mostly about that failure: an action with a novel type must
still render correctly from `kind` alone.
"""
import logging

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data.base import MarketData
from beacon.data.corporate_actions import (
    ACTION_TYPES,
    CASH,
    CASH_ACTIONS,
    KINDS,
    RATIO,
    RATIO_ACTIONS,
    STATUSES,
    STRUCTURAL,
    STRUCTURAL_ACTIONS,
    CorporateActions,
    kind_of,
    status_of,
)
from beacon.data.fetcher import DataFetcher
from beacon.server import ServerConfig, create_app

TOKEN = "test-token-value"
DATES = pd.bdate_range("2025-01-02", periods=40)


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def build_client(actions: pd.DataFrame) -> TestClient:
    """A server over one asset and a supplied action history."""
    market = MarketData.from_dataframe(pd.DataFrame([
        {"IDENTIFIER": "AAA", "DATE": date, "CLOSE": 100.0 + index}
        for index, date in enumerate(DATES)]))

    fetcher = DataFetcher(market, None, CorporateActions.from_dataframe(actions))

    return TestClient(create_app(
        ServerConfig(auth_token=TOKEN, data_fetcher=fetcher)))


class TestKindOf:
    """The engine's answer to what `value` means."""

    @pytest.mark.parametrize("action_type", sorted(CASH_ACTIONS))
    def test_cash_actions(self, action_type):
        assert kind_of(action_type) == CASH

    @pytest.mark.parametrize("action_type", sorted(RATIO_ACTIONS))
    def test_ratio_actions(self, action_type):
        assert kind_of(action_type) == RATIO

    @pytest.mark.parametrize("action_type", sorted(STRUCTURAL_ACTIONS))
    def test_structural_actions(self, action_type):
        """The issue asked for a cash/ratio pair. The library has had three
        categories since actions were added, and a rights issue is neither: it
        carries no directly aggregable value, so calling it cash or ratio would
        be a lie that renders as a number."""
        assert kind_of(action_type) == STRUCTURAL

    def test_every_known_type_has_a_kind(self):
        assert all(kind_of(action_type) in KINDS for action_type in ACTION_TYPES)

    def test_the_three_sets_do_not_overlap(self):
        """If a type were in two sets, its kind would depend on iteration
        order — which is exactly the kind of bug nothing would surface."""
        assert not CASH_ACTIONS & RATIO_ACTIONS
        assert not CASH_ACTIONS & STRUCTURAL_ACTIONS
        assert not RATIO_ACTIONS & STRUCTURAL_ACTIONS

    def test_it_is_case_insensitive(self):
        assert kind_of("dividend") == CASH

    def test_an_unknown_type_is_structural_not_cash(self, caplog):
        """The safe direction to be wrong in: structural makes a client show no
        quantity, where cash would put a misinterpreted number on screen."""
        with caplog.at_level(logging.WARNING):
            assert kind_of("SOMETHING_NEW") == STRUCTURAL

        assert "Unrecognised action type" in caplog.text


class TestStatusOf:
    """Statuses are validated, not passed through."""

    @pytest.mark.parametrize("status", STATUSES)
    def test_known_statuses_survive(self, status):
        assert status_of(status) == status

    def test_it_normalises_case_and_padding(self):
        assert status_of("  Paid ") == "paid"

    def test_none_is_none(self):
        assert status_of(None) is None
        assert status_of(float("nan")) is None

    def test_an_unknown_status_is_unknown_rather_than_forwarded(self, caplog):
        """Forwarding it would make a client branch on a value no schema
        documents."""
        with caplog.at_level(logging.WARNING):
            assert status_of("pending-review") is None

        assert "Unrecognised action status" in caplog.text


class TestServedActions:
    """What the endpoint puts on the wire."""

    @pytest.fixture
    def client(self) -> TestClient:
        return build_client(pd.DataFrame([
            {"IDENTIFIER": "AAA", "EX_DATE": DATES[3], "TYPE": "DIVIDEND",
             "VALUE": 0.25, "PAY_DATE": DATES[10], "STATUS": "paid"},
            {"IDENTIFIER": "AAA", "EX_DATE": DATES[8], "TYPE": "SPLIT",
             "VALUE": 2.0, "PAY_DATE": DATES[8], "STATUS": "paid"},
            {"IDENTIFIER": "AAA", "EX_DATE": DATES[20], "TYPE": "SPIN_OFF",
             "VALUE": 0.4, "PAY_DATE": None, "STATUS": "announced"},
        ]))

    def actions(self, client) -> dict[str, dict]:
        body = client.get("/data/corporate-actions/AAA", headers=auth()).json()

        return {action["type"]: action for action in body["actions"]}

    def test_a_dividend_is_cash(self, client):
        assert self.actions(client)["DIVIDEND"]["kind"] == CASH

    def test_a_split_is_a_ratio(self, client):
        assert self.actions(client)["SPLIT"]["kind"] == RATIO

    def test_a_spin_off_is_structural(self, client):
        assert self.actions(client)["SPIN_OFF"]["kind"] == STRUCTURAL

    def test_pay_date_and_status_are_served(self, client):
        dividend = self.actions(client)["DIVIDEND"]

        assert dividend["pay_date"] == str(DATES[10].date())
        assert dividend["status"] == "paid"

    def test_an_unknown_pay_date_is_null_not_a_dash(self, client):
        """Null so the client omits the field. A dash reads as "there is
        none", which is a different statement from "we do not know"."""
        spin_off = self.actions(client)["SPIN_OFF"]

        assert spin_off["pay_date"] is None
        assert spin_off["status"] == "announced"

    def test_every_action_carries_a_kind(self, client):
        body = client.get("/data/corporate-actions/AAA", headers=auth()).json()

        assert all(action["kind"] in KINDS for action in body["actions"])


class TestWithoutTheOptionalColumns:
    """A history that has never heard of pay dates still serves."""

    @pytest.fixture
    def client(self) -> TestClient:
        return build_client(pd.DataFrame([
            {"IDENTIFIER": "AAA", "EX_DATE": DATES[3], "TYPE": "DIVIDEND",
             "VALUE": 0.25},
        ]))

    def test_kind_is_still_derived(self, client):
        """Derived from the type rather than stored, so an older store gains
        the field for free."""
        body = client.get("/data/corporate-actions/AAA", headers=auth()).json()

        assert body["actions"][0]["kind"] == CASH

    def test_the_absent_columns_are_null(self, client):
        body = client.get("/data/corporate-actions/AAA", headers=auth()).json()
        action = body["actions"][0]

        assert action["pay_date"] is None
        assert action["status"] is None


class TestTheClientNeedsNoTypeList:
    """The acceptance criterion, stated as a test."""

    def test_a_novel_ratio_type_renders_by_kind_alone(self):
        """A client keying off `kind` handles a type it has never seen. One
        keying off a type-string list renders this as a cash amount."""
        client = build_client(pd.DataFrame([
            {"IDENTIFIER": "AAA", "EX_DATE": DATES[5], "TYPE": "STOCK_DIVIDEND",
             "VALUE": 1.05},
        ]))

        body = client.get("/data/corporate-actions/AAA", headers=auth()).json()
        action = body["actions"][0]

        assert action["type"] == "STOCK_DIVIDEND"
        assert action["kind"] == RATIO
        assert action["value"] == pytest.approx(1.05)


class TestGeneratedActions:
    """BN-114's generator populates the new fields."""

    def test_dividends_settle_after_their_ex_date(self):
        from beacon.synthetic import SyntheticConfig, generate

        dataset = generate(SyntheticConfig(assets=12, start="2022-01-03",
                                           end="2024-06-28", seed=5))
        actions = dataset.actions.data.reset_index(drop=True)
        dividends = actions[actions["TYPE"] == "DIVIDEND"]

        assert (dividends["PAY_DATE"] > dividends["EX_DATE"]).all()

    def test_a_split_settles_on_its_ex_date(self):
        from beacon.synthetic import SyntheticConfig, generate

        dataset = generate(SyntheticConfig(assets=60, start="2019-12-31",
                                           end="2024-12-31", seed=5))
        actions = dataset.actions.data.reset_index(drop=True)
        splits = actions[actions["TYPE"] == "SPLIT"]

        assert not splits.empty
        assert (splits["PAY_DATE"] == splits["EX_DATE"]).all()

    def test_status_follows_from_whether_the_pay_date_arrived(self):
        """A panel ending days after an ex-date leaves that dividend announced
        rather than paid — the state is derived, not stamped."""
        from beacon.synthetic import SyntheticConfig, generate

        dataset = generate(SyntheticConfig(assets=12, start="2022-01-03",
                                           end="2024-08-20", seed=3))
        actions = dataset.actions.data.reset_index(drop=True)

        assert set(actions["STATUS"]) == {"paid", "announced"}

        announced = actions[actions["STATUS"] == "announced"]
        assert (announced["PAY_DATE"] > pd.Timestamp("2024-08-20")).all()

    def test_pay_dates_survive_a_store_round_trip(self, tmp_path):
        """They are written and read as text, so the type has to be restored
        or the API would serialise a string where a date is documented."""
        from beacon.data import store
        from beacon.synthetic import SyntheticConfig, generate

        dataset = generate(SyntheticConfig(assets=8, start="2022-01-03",
                                           end="2023-06-30", seed=2))
        path = store.save(dataset.fetcher(), tmp_path / "store")

        actions = store.load(path).corporate_actions.data

        assert pd.api.types.is_datetime64_any_dtype(actions["PAY_DATE"])
        assert set(actions["STATUS"]) <= set(STATUSES)
