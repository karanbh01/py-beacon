# tests/test_identifier_search.py
"""BN-127: identifier search and enumeration.

The acceptance list in the client's spec is reproduced here case for case, with
its own example symbols, so a reader can put the two side by side rather than
translating.

Two properties carry the endpoint. **Ranking is the server's** — once `limit`
is applied the client cannot re-rank what it was not sent — and **an empty
store answers 200**, because "nothing matches" and "this engine is
misconfigured" render as very different things and an empty suggestion list
must not look like a broken install.
"""
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data.base import MarketData, ReferenceData
from beacon.data.corporate_actions import CorporateActions
from beacon.data.fetcher import DataFetcher
from beacon.data.identifiers import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    NO_MATCH,
    IdentifierEntry,
    IdentifierIndex,
)
from beacon.server import ServerConfig, create_app

TOKEN = "test-token-value"
DATES = pd.bdate_range("2025-01-02", periods=5)

# The spec's own examples, so its acceptance list reads directly against this.
# NOTREF is priced but has no reference row; REFONLY is the reverse.
PRICED = ["AAPL", "AAP", "MSFT", "CMP000", "CMP001", "NOTREF"]
NAMED = {
    "AAPL": "Apple Inc.",
    "AAP": "Advance Auto Parts",
    "MSFT": "Microsoft Corporation",
    "CMP000": "Zeta Holdings",
    "CMP001": "Apple Growers Co-operative",
    "REFONLY": "Reference Only Limited",
}


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def build_fetcher() -> DataFetcher:
    """Prices for six names, reference for six, overlapping in four."""
    market = MarketData.from_dataframe(pd.DataFrame([
        {"IDENTIFIER": name, "DATE": date, "CLOSE": 100.0}
        for name in PRICED for date in DATES]))

    reference = ReferenceData.from_dataframe(pd.DataFrame([
        {"IDENTIFIER": name, "DATE_FROM": "2020-01-01", "NAME": display,
         "EXCHANGE": "XNAS", "CURRENCY": "USD"}
        for name, display in NAMED.items()]))

    actions = CorporateActions.from_dataframe(pd.DataFrame([
        {"IDENTIFIER": "AAPL", "EX_DATE": DATES[1], "TYPE": "DIVIDEND",
         "VALUE": 0.25}]))

    return DataFetcher(market, reference, actions)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(
        ServerConfig(auth_token=TOKEN, data_fetcher=build_fetcher())))


def search(client: TestClient,
           **params) -> dict:
    """GET the endpoint and return the decoded body."""
    response = client.get("/data/identifiers", params=params, headers=auth())

    assert response.status_code == 200, response.text

    return response.json()


def coverage_is_tracing() -> bool:
    """Whether coverage instrumentation is active.

    A timing assertion made under tracing measures the instrumentation rather
    than the code — the same search that takes 22ms bare takes 65ms traced. So
    the performance test is skipped when coverage is on rather than having its
    threshold loosened to accommodate it, which would leave a number that
    asserts nothing in either mode.
    """
    try:
        import coverage
    except ImportError:
        return False

    return coverage.Coverage.current() is not None


def names_of(body: dict) -> list[str]:
    """The identifiers a response returned, in order."""
    return [row["identifier"] for row in body["identifiers"]]


class TestAcceptance:
    """The client's list in §9, case for case."""

    def test_q_aapl_returns_aapl_first(self, client):
        assert names_of(search(client, q="aapl"))[0] == "AAPL"

    def test_q_apple_finds_aapl_by_name(self, client):
        """Matching runs against name as well as identifier."""
        assert "AAPL" in names_of(search(client, q="apple"))

    def test_identifier_prefix_beats_name_match(self, client):
        """`CMP000` before `CMP001`, and both before anything matched only on
        its name. Someone typing `cmp00` wants the ticker, not the first
        company whose description happens to contain the fragment."""
        found = names_of(search(client, q="cmp00"))

        assert found[:2] == ["CMP000", "CMP001"]

    def test_absent_q_enumerates(self, client):
        body = search(client)

        assert len(body["identifiers"]) == min(DEFAULT_LIMIT, body["total"])
        assert body["total"] == len(PRICED) + 1   # REFONLY is priced nowhere

    def test_limit_bounds_rows_but_not_total(self, client):
        body = search(client, limit=2)

        assert len(body["identifiers"]) == 2
        assert body["truncated"] is True
        assert body["total"] == 7

    def test_an_identifier_with_no_name_is_returned_not_dropped(self, client):
        """A row without a name is still a useful suggestion."""
        body = search(client, q="notref")
        rows = {row["identifier"]: row for row in body["identifiers"]}

        assert "NOTREF" in rows
        assert rows["NOTREF"]["name"] is None

    def test_datasets_market_excludes_a_reference_only_name(self, client):
        assert "REFONLY" not in names_of(search(client, datasets="market",
                                                limit=MAX_LIMIT))
        assert "REFONLY" in names_of(search(client, limit=MAX_LIMIT))

    def test_an_empty_store_is_200_with_nothing(self):
        """Not a 404 and not CONFIGURATION_ERROR. An empty search must not
        look like a broken install."""
        client = TestClient(create_app(ServerConfig(auth_token=TOKEN)))
        response = client.get("/data/identifiers", headers=auth())

        assert response.status_code == 200
        assert response.json()["identifiers"] == []
        assert response.json()["total"] == 0

    def test_no_match_is_200_not_404(self, client):
        body = search(client, q="zzzznotathing")

        assert body["identifiers"] == []
        assert body["total"] == 0
        assert body["truncated"] is False

    @pytest.mark.skipif(coverage_is_tracing(),
                        reason="timing under coverage measures the tracer")
    def test_it_is_fast_enough_for_a_keystroke(self):
        """p99 under 50ms at limit=20. Above ~150ms the suggestions arrive
        after the next keystroke and the list visibly lags the field.

        Run this bare: `pytest tests/test_identifier_search.py --no-cov`.
        """
        entries = []
        for position in range(100_000):
            identifier = f"SYM{position:06d}"
            name = f"Company {position:06d}"
            entries.append(IdentifierEntry(
                identifier, name, ("market", "reference"), "XNAS", "USD",
                identifier.casefold(), name.casefold()))

        index = IdentifierIndex(tuple(entries), "bench")

        durations = []
        for query in ("SYM000001", "sym0001", "company 0", "zzz") * 25:
            started = time.perf_counter()
            index.search(query, limit=20)
            durations.append((time.perf_counter() - started) * 1000)

        durations.sort()
        p99 = durations[int(len(durations) * 0.99) - 1]

        assert p99 < 50.0, f"p99 {p99:.1f}ms over 100k identifiers"


class TestRanking:
    """The tier order, stated as a contract."""

    def test_exact_beats_prefix(self, client):
        """AAP is a prefix of AAPL, so a query of `aap` matches both — the
        exact one must come first."""
        assert names_of(search(client, q="aap"))[0] == "AAP"

    def test_identifier_prefix_beats_name_prefix(self, client):
        """A query both tiers match: `AAP` and `AAPL` start with "a", and
        `CMP001` is 'Apple Growers', whose *name* does. The identifiers win."""
        found = names_of(search(client, q="a", limit=MAX_LIMIT))

        assert found.index("AAPL") < found.index("CMP001")

    def test_name_prefix_beats_identifier_substring(self, client):
        """`AAP` contains "ap" without starting with it, while 'Apple Inc.'
        and 'Apple Growers' both start with it. The names win.

        Worth noting the near-miss: "appl" does *not* demonstrate this, because
        AAPL's own name starts with it too, so both rows land in the same tier
        and the order is the alphabetical tie-break rather than the rule."""
        found = names_of(search(client, q="ap", limit=MAX_LIMIT))

        assert found.index("CMP001") < found.index("AAP")
        assert found.index("AAPL") < found.index("AAP")

    def test_matching_is_case_insensitive(self, client):
        assert names_of(search(client, q="AaPl")) == names_of(search(client, q="aapl"))

    def test_ties_break_alphabetically(self, client):
        """Within a tier, order is stable and predictable rather than whatever
        the store happened to hold."""
        found = names_of(search(client, q="cmp", limit=MAX_LIMIT))

        assert found == sorted(found)

    def test_the_readable_rule_and_the_fast_loop_agree(self):
        """`rank_against` states the rule; `search` inlines it for speed. A
        divergence between the two would be invisible until the ordering was
        wrong, so they are pinned together."""
        entries = [
            IdentifierEntry("ABC", "Alpha", ("market",), None, None, "abc", "alpha"),
            IdentifierEntry("XABC", "Beta", ("market",), None, None, "xabc", "beta"),
            IdentifierEntry("ZZZ", "Abc Holdings", ("market",), None, None,
                            "zzz", "abc holdings"),
            IdentifierEntry("QQQ", "Quite Abc", ("market",), None, None,
                            "qqq", "quite abc"),
            IdentifierEntry("NNN", "Nothing", ("market",), None, None,
                            "nnn", "nothing"),
        ]
        index = IdentifierIndex(tuple(sorted(entries, key=lambda e: e.identifier)),
                                "v")

        expected = sorted(
            [entry for entry in entries if entry.rank_against("abc") != NO_MATCH],
            key=lambda entry: (entry.rank_against("abc"), entry.identifier))

        found = index.search("abc", limit=MAX_LIMIT)

        assert [entry.identifier for entry in found.entries] == [
            entry.identifier for entry in expected]


class TestPaging:
    """Walking a full enumeration."""

    def test_offset_skips(self, client):
        everything = names_of(search(client, limit=MAX_LIMIT))
        skipped = names_of(search(client, offset=2, limit=MAX_LIMIT))

        assert skipped == everything[2:]

    def test_offset_works_within_a_search(self, client):
        everything = names_of(search(client, q="cmp", limit=MAX_LIMIT))
        skipped = names_of(search(client, q="cmp", offset=1, limit=MAX_LIMIT))

        assert skipped == everything[1:]

    def test_the_window_spans_tiers(self, client):
        """A limit that runs past the end of one tier continues into the next
        rather than stopping."""
        one = names_of(search(client, q="aap", limit=1))
        two = names_of(search(client, q="aap", limit=2))

        assert two[0] == one[0]
        assert len(two) == 2

    def test_an_offset_past_the_end_is_empty_not_an_error(self, client):
        body = search(client, offset=10_000)

        assert body["identifiers"] == []
        assert body["total"] == 7
        assert body["truncated"] is False

    def test_the_limit_is_capped(self, client):
        response = client.get("/data/identifiers",
                              params={"limit": MAX_LIMIT + 1}, headers=auth())

        assert response.status_code == 422

    def test_a_zero_limit_is_refused(self, client):
        assert client.get("/data/identifiers", params={"limit": 0},
                          headers=auth()).status_code == 422


class TestCoverage:
    """`datasets` is what stops the UI offering a ticker it cannot chart."""

    def test_each_row_reports_what_covers_it(self, client):
        rows = {row["identifier"]: row for row in
                search(client, limit=MAX_LIMIT)["identifiers"]}

        assert set(rows["AAPL"]["datasets"]) == {"market", "reference",
                                                 "corporate_actions"}
        assert rows["NOTREF"]["datasets"] == ["market"]
        assert rows["REFONLY"]["datasets"] == ["reference"]

    def test_filtering_requires_all_named_datasets(self, client):
        """Intersection, not union: asking for market *and* corporate_actions
        means both."""
        found = names_of(search(client, datasets="market,corporate_actions",
                                limit=MAX_LIMIT))

        assert found == ["AAPL"]

    def test_an_unknown_dataset_matches_nothing(self, client):
        assert search(client, datasets="nonsense")["total"] == 0

    def test_the_extras_are_carried_when_present(self, client):
        row = {r["identifier"]: r for r in
               search(client, q="aapl")["identifiers"]}["AAPL"]

        assert row["exchange"] == "XNAS"
        assert row["currency"] == "USD"


class TestCaching:
    """An ETag the client can revalidate against."""

    def test_an_etag_is_served(self, client):
        response = client.get("/data/identifiers", headers=auth())

        assert response.headers.get("ETag")

    def test_the_version_matches_the_etag(self, client):
        response = client.get("/data/identifiers", headers=auth())

        assert response.json()["version"] in response.headers["ETag"]

    def test_it_is_stable_across_requests(self, client):
        first = client.get("/data/identifiers", headers=auth())
        second = client.get("/data/identifiers", headers=auth())

        assert first.headers["ETag"] == second.headers["ETag"]

    def test_it_moves_when_data_is_merged(self):
        """A sync records a refresh, which is what the fingerprint reads — so
        suggestions invalidate with the freshness event a client already
        listens for, and need no new mechanism."""
        fetcher = build_fetcher()
        client = TestClient(create_app(
            ServerConfig(auth_token=TOKEN, data_fetcher=fetcher)))

        before = client.get("/data/identifiers", headers=auth()).headers["ETag"]

        fetcher.merge_market_data(pd.DataFrame([
            {"IDENTIFIER": "NEWNAME", "DATE": DATES[0], "CLOSE": 10.0}]))

        after = client.get("/data/identifiers", headers=auth())

        assert after.headers["ETag"] != before
        assert "NEWNAME" in names_of(after.json())

    def test_the_index_is_not_rebuilt_per_request(self):
        """The whole point of caching it. Rebuilding pulls names out of pandas,
        which is the expensive part and only changes when the data does."""
        fetcher = build_fetcher()
        client = TestClient(create_app(
            ServerConfig(auth_token=TOKEN, data_fetcher=fetcher)))

        builds = []
        original = IdentifierIndex.build

        def counting(source):
            builds.append(1)

            return original(source)

        IdentifierIndex.build = staticmethod(counting)
        try:
            for _ in range(5):
                client.get("/data/identifiers", params={"q": "aap"},
                           headers=auth())
        finally:
            IdentifierIndex.build = original

        assert len(builds) == 1, f"rebuilt {len(builds)} times"


class TestContract:
    """What the generated client depends on."""

    def test_it_appears_in_the_openapi_schema(self, client):
        schema = client.get("/openapi.json", headers=auth()).json()

        assert "/data/identifiers" in schema["paths"]

    def test_the_response_has_a_real_schema(self, client):
        """beacon-ui generates its typed client from this. A response typed as
        `additionalProperties: true` is untyped as far as the generator is
        concerned."""
        schema = client.get("/openapi.json", headers=auth()).json()
        response = (schema["paths"]["/data/identifiers"]["get"]["responses"]
                    ["200"]["content"]["application/json"]["schema"])

        assert "$ref" in response or response.get("type") == "object"

        model = schema["components"]["schemas"]["IdentifierSearchResponse"]
        assert set(model["properties"]) >= {"identifiers", "total", "truncated"}

        row = schema["components"]["schemas"]["IdentifierMatch"]
        assert set(row["properties"]) >= {"identifier", "name", "datasets"}

    def test_the_query_parameters_are_declared(self, client):
        schema = client.get("/openapi.json", headers=auth()).json()
        declared = {parameter["name"] for parameter in
                    schema["paths"]["/data/identifiers"]["get"]["parameters"]}

        assert {"q", "limit", "offset", "datasets"} <= declared

    def test_it_requires_authentication(self, client):
        assert client.get("/data/identifiers").status_code == 401


class TestIndexConstruction:
    """Building the index off a fetcher."""

    def test_it_covers_the_union_of_every_dataset(self):
        index = IdentifierIndex.build(build_fetcher())

        assert {entry.identifier for entry in index.entries} == set(
            PRICED) | set(NAMED)

    def test_entries_come_out_alphabetical(self):
        """Which is what makes the tier buckets sorted for free."""
        index = IdentifierIndex.build(build_fetcher())
        found = [entry.identifier for entry in index.entries]

        assert found == sorted(found)

    def test_folded_forms_are_precomputed(self):
        """Folding per request is what makes a 100k search miss its budget."""
        index = IdentifierIndex.build(build_fetcher())

        for entry in index.entries:
            assert entry.folded_identifier == entry.identifier.casefold()

    def test_an_empty_index_searches_without_error(self):
        result = IdentifierIndex.empty().search("anything")

        assert result.entries == ()
        assert result.total == 0

    def test_a_fetcher_without_reference_data_still_indexes(self):
        """Names are optional; identifiers are not."""
        market = MarketData.from_dataframe(pd.DataFrame([
            {"IDENTIFIER": "AAA", "DATE": DATES[0], "CLOSE": 1.0}]))
        index = IdentifierIndex.build(DataFetcher(market))

        assert [entry.identifier for entry in index.entries] == ["AAA"]
        assert index.entries[0].name is None

    def test_a_blank_reference_name_reads_as_absent(self):
        """Rather than as an empty-string name, which would match every query
        as a prefix."""
        market = MarketData.from_dataframe(pd.DataFrame([
            {"IDENTIFIER": "AAA", "DATE": DATES[0], "CLOSE": 1.0}]))
        reference = ReferenceData.from_dataframe(pd.DataFrame([
            {"IDENTIFIER": "AAA", "DATE_FROM": "2020-01-01", "NAME": "  "}]))

        index = IdentifierIndex.build(DataFetcher(market, reference))

        assert index.entries[0].name is None
        assert index.search("zzz").total == 0
