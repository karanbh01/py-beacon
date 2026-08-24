# tests/test_universe_filters.py
"""BN-143: expressions in universe filters and backtest screens.

The acceptance cases are equivalence — a filtered universe must contain what
the equivalent explicit list contains — and the frozen/live distinction, which
is the part a user has to be *told* rather than left to discover.
"""
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon import universe
from beacon.backtest.engine import TradeInstruction
from beacon.backtest.rules import ExpressionScreen
from beacon.expressions import data
from beacon.portfolio.base import Portfolio
from beacon.server import ServerConfig, create_app
from beacon.testing import dataset

TOKEN = "universe-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
REBALANCE = pd.Timestamp("2024-06-03")


@pytest.fixture(scope="module")
def fetcher():
    return dataset.data_fetcher()


@pytest.fixture
def client(fetcher):
    return TestClient(create_app(ServerConfig(
        auth_token=TOKEN, data_fetcher=fetcher,
        storage_root=Path(tempfile.mkdtemp()))))


class TestWhere:
    """`universe.where(expression)`."""

    def test_it_selects_the_matching_names(self,
                                           fetcher):
        members = universe.where(data.market.close > 0, fetcher)

        assert set(members) == set(fetcher.reference_identifiers)

    def test_a_narrower_filter_selects_fewer(self,
                                             fetcher):
        wide = universe.where(data.market.close > 0, fetcher)
        narrow = universe.where(data.market.close > 1e9, fetcher)

        assert len(narrow) < len(wide)

    def test_it_composes(self,
                         fetcher):
        members = universe.where(
            (data.market.close > 0) & (data.reference.sector == "Technology"),
            fetcher)
        sectors = {universe_member: fetcher.fetch_reference_data(
            universe_member, "2024-06-03")["SECTOR"].iloc[0]
            for universe_member in members}

        assert set(sectors.values()) == {"Technology"}

    def test_fx_pairs_are_not_universe_members(self,
                                               fetcher):
        """The market data carries an identifier per FX pair (`GBPUSD` here),
        which would otherwise be offered as an instrument somebody could
        hold."""
        members = universe.where(data.market.close > 0, fetcher)

        assert "GBPUSD" in fetcher.identifiers
        assert "GBPUSD" not in members

    def test_the_order_is_stable(self,
                                 fetcher):
        """Two runs over the same data must produce the same list, or a
        universe document changes when nothing did."""
        first = universe.where(data.market.close > 0, fetcher)
        second = universe.where(data.market.close > 0, fetcher)

        assert first == second

    def test_it_resolves_as_of_a_date(self,
                                      fetcher):
        """A universe built as of a date contains what was knowable then, the
        same guarantee the index rules give.

        Tested against a date before the panel starts, where the right answer
        is *nothing*: every name has no price knowable yet. Comparing two
        in-range dates would not do — the membership happens to be the same on
        both, so that version passed whether or not the date was used at all.
        """
        before_the_data = universe.where(data.market.close > 0, fetcher,
                                         "2022-01-01")
        within = universe.where(data.market.close > 0, fetcher, "2024-06-03")

        assert before_the_data == []
        assert within

    def test_candidates_can_be_narrowed(self,
                                        fetcher):
        members = universe.where(data.market.close > 0, fetcher,
                                 identifiers=["AAA", "BBB"])

        assert set(members) <= {"AAA", "BBB"}


class TestFrozenAndLive:
    """Different objects, and a user needs to know which they have."""

    def test_build_keeps_both_the_question_and_the_answer(self,
                                                          fetcher):
        built = universe.build(data.market.close > 0, fetcher)

        assert built.identifiers
        assert built.expression.to_dict()["node"] == "comparison"

    def test_a_live_universe_re_evaluates(self,
                                          fetcher):
        document = universe.build(data.market.close > 1e9, fetcher,
                                  mode=universe.LIVE).as_document()

        # The stored membership says one thing; the filter says another. A
        # live universe must answer with the filter.
        document["identifiers"] = ["AAA"]

        assert universe.resolve_document(document, fetcher) != ["AAA"]

    def test_a_frozen_universe_does_not(self,
                                        fetcher):
        document = universe.build(data.market.close > 0, fetcher,
                                  mode=universe.FROZEN).as_document()
        document["identifiers"] = ["AAA"]

        assert universe.resolve_document(document, fetcher) == ["AAA"]

    def test_a_document_without_a_filter_is_frozen(self,
                                                   fetcher):
        """A curated list has no question to re-ask, so the absence of a
        filter decides regardless of what `mode` says."""
        document = {"identifiers": ["AAA", "BBB"], "mode": universe.LIVE}

        assert universe.resolve_document(document, fetcher) == ["AAA", "BBB"]

    def test_the_document_records_which_it_is(self,
                                              fetcher):
        document = universe.build(data.market.close > 0, fetcher,
                                  mode=universe.LIVE).as_document()

        assert document["mode"] == universe.LIVE
        assert document["as_of"]

    def test_an_unknown_mode_is_refused(self,
                                        fetcher):
        with pytest.raises(ValueError, match="mode must be"):
            universe.build(data.market.close > 0, fetcher, mode="sometimes")


class TestTheApi:
    """`POST /universes` with a filter."""

    def created(self,
                client,
                **body):
        return client.post("/universes", headers=HEADERS, json=body)

    def test_a_filter_builds_the_same_membership_as_the_list(self,
                                                             client,
                                                             fetcher):
        """The acceptance case: a filtered universe contains what the
        equivalent explicit call contains."""
        expected = universe.where(data.market.close > 0, fetcher)

        filtered = self.created(client, name="Filtered",
                                filter=(data.market.close > 0).to_dict())
        listed = self.created(client, name="Listed", identifiers=expected)

        assert filtered.status_code == 201
        assert set(filtered.json()["identifiers"]) == set(
            listed.json()["identifiers"])

    def test_the_document_carries_its_filter(self,
                                             client):
        """So a filtered universe records *how* it was built. One that is only
        a list cannot be refreshed when the data moves."""
        created = self.created(client, name="Carried",
                               filter=(data.market.close > 0).to_dict())

        assert created.json()["filter"]["node"] == "comparison"
        assert created.json()["as_of"]

    def test_a_live_universe_re_evaluates_on_read(self,
                                                  client):
        created = self.created(client, name="Live", mode="live",
                               filter=(data.market.close > 0).to_dict())
        members = client.get(f"/universes/{created.json()['id']}/members",
                             headers=HEADERS)

        assert set(members.json()["identifiers"]) == set(
            created.json()["identifiers"])

    def test_a_live_universe_answers_as_of_a_date(self,
                                                  client):
        created = self.created(client, name="Dated", mode="live",
                               filter=(data.market.close > 0).to_dict())
        members = client.get(f"/universes/{created.json()['id']}/members",
                             headers=HEADERS, params={"date": "2024-02-01"})

        assert members.status_code == 200

    def test_both_a_filter_and_a_list_is_refused(self,
                                                 client):
        """Ambiguous about which is the definition. Guessing would make the
        answer depend on an implementation detail."""
        response = self.created(client, name="Both", identifiers=["AAA"],
                                filter=(data.market.close > 0).to_dict())

        assert response.status_code == 422

        findings = response.json()["error"]["detail"]["findings"]

        assert any(finding["code"] == "AMBIGUOUS_MEMBERSHIP"
                   for finding in findings)

    def test_a_filter_matching_nothing_is_refused(self,
                                                  client):
        """An empty universe is not a thing anybody meant to make, and a
        filter that silently produces one is the failure this whole layer
        exists to prevent."""
        response = self.created(client, name="Empty",
                                filter=(data.market.close > 1e30).to_dict())

        assert response.status_code == 422
        assert any(finding["code"] == "EMPTY_FILTER" for finding
                   in response.json()["error"]["detail"]["findings"])

    def test_a_malformed_filter_is_refused(self,
                                           client):
        response = self.created(client, name="Broken",
                                filter={"node": "sometimes"})

        assert response.status_code == 422

    def test_an_unknown_mode_is_refused(self,
                                        client):
        response = self.created(client, name="Odd", mode="sometimes",
                                filter=(data.market.close > 0).to_dict())

        assert response.status_code == 422

    def test_a_curated_universe_is_unchanged(self,
                                             client):
        """The existing shape still works, and reports itself as frozen."""
        created = self.created(client, name="Curated",
                               identifiers=["AAA", "BBB"])

        assert created.json()["mode"] == "frozen"
        assert created.json()["filter"] is None


class TestTheBacktestScreen:
    """`ExpressionScreen`, composed with the modifier chain."""

    def portfolio_of(self,
                     holdings: dict[str, float]) -> Portfolio:
        portfolio = Portfolio(portfolio_id="screen-test",
                              initial_cash=1_000_000.0)

        for asset_id, quantity in holdings.items():
            portfolio.execute_buy(asset_id, quantity, 100.0, date=REBALANCE)

        return portfolio

    def test_it_never_skips_a_rebalance(self,
                                        fetcher):
        """A screen changes what is traded, not whether."""
        screen = ExpressionScreen(data.market.close > 0, fetcher)

        assert not screen.should_skip_rebalance(REBALANCE,
                                                self.portfolio_of({}), {})

    def test_it_drops_buys_of_excluded_names(self,
                                             fetcher):
        screen = ExpressionScreen(data.market.close > 1e9, fetcher)
        trades = [TradeInstruction("AAA", "BUY", 10.0, 100.0, 0.0)]

        assert screen.adjust_trades(trades, REBALANCE,
                                    self.portfolio_of({})) == []

    def test_it_keeps_buys_of_passing_names(self,
                                            fetcher):
        screen = ExpressionScreen(data.market.close > 0, fetcher)
        trades = [TradeInstruction("AAA", "BUY", 10.0, 100.0, 0.0)]

        assert screen.adjust_trades(trades, REBALANCE,
                                    self.portfolio_of({})) == trades

    def test_it_sells_a_holding_that_fails(self,
                                           fetcher):
        """Dropping only the buys would leave the position from before the
        screen turned against it — so the screen would appear to work on new
        entries and quietly not apply to anything held."""
        screen = ExpressionScreen(data.market.close > 1e9, fetcher)
        portfolio = self.portfolio_of({"AAA": 10.0})

        adjusted = screen.adjust_trades([], REBALANCE, portfolio)

        assert [(trade.asset_id, trade.side) for trade in adjusted] == [
            ("AAA", "SELL")]

    def test_it_does_not_duplicate_an_existing_sell(self,
                                                    fetcher):
        screen = ExpressionScreen(data.market.close > 1e9, fetcher)
        portfolio = self.portfolio_of({"AAA": 10.0})
        trades = [TradeInstruction("AAA", "SELL", 10.0, 100.0, 0.0)]

        adjusted = screen.adjust_trades(trades, REBALANCE, portfolio)

        assert len(adjusted) == 1

    def test_it_is_re_evaluated_per_rebalance(self,
                                              fetcher):
        """Resolving once at the start would let a name that only became
        liquid later pass an earlier rebalance — look-ahead wearing a
        different hat, and it makes the backtest look better rather than
        fail."""
        screen = ExpressionScreen(data.market.close > 0, fetcher)
        portfolio = self.portfolio_of({})
        trades = [TradeInstruction("AAA", "BUY", 10.0, 100.0, 0.0)]

        early = screen.adjust_trades(trades, pd.Timestamp("2024-02-01"),
                                     portfolio)
        late = screen.adjust_trades(trades, REBALANCE, portfolio)

        assert early == trades and late == trades

    def test_it_drops_the_same_names_an_index_rule_would(self,
                                                         fetcher):
        """The acceptance case: a backtest screened by an expression drops
        what the equivalent index rule drops."""
        from beacon.asset.equity import Equity
        from beacon.index import ExpressionRule

        expression = data.market.market_cap > 1e12
        screen = ExpressionScreen(expression, fetcher)
        rule = ExpressionRule.from_expression(expression)

        for name in ["AAA", "BBB", "CCC", "DDD"]:
            asset = Equity(asset_id=name, ticker=name, name=name,
                           currency="USD", exchange="XNYS")

            assert screen.passes(name, REBALANCE) == rule.is_eligible(
                asset, REBALANCE, fetcher)
