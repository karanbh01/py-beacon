# tests/test_server_universes.py
"""BN-132: creating universes, and the GLOBAL one every dataset gets.

The router had `GET`, `PUT` and `DELETE` but no `POST`: a universe could be
mutated if it already existed and never brought into being. Nothing seeded one
either, so a fresh workspace answered `GET /universes` with an empty list —
all the data present and none of it selectable.

**Members are checked against the loaded data.** A universe naming an
instrument the server has no prices for produces an empty index and no
explanation, so both write paths resolve every member and refuse the ones that
are missing *by name*. Telling somebody a list of five hundred tickers is
wrong without saying which one is not an error message.
"""
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from beacon.server.routers.universes import GLOBAL_ID, slug
from beacon.testing import dataset

TOKEN = "universe-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client():
    """A fresh server, with its own storage root per test.

    Function-scoped rather than shared: these tests create and delete
    documents, and a shared store would make them order-dependent.
    """
    app = create_app(ServerConfig(auth_token=TOKEN,
                                  data_fetcher=dataset.data_fetcher(),
                                  storage_root=Path(tempfile.mkdtemp())))

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def known():
    """The instruments a universe may contain.

    Reference identifiers, not market ones. Since BN-128 the market data also
    carries a row set per FX pair -- `GBPUSD` and friends -- because that is
    how `fetch_fx_rates` finds them. They are not companies and cannot be
    index constituents, so a universe built from `fetcher.identifiers` would
    offer somebody a currency pair to select.
    """
    return dataset.data_fetcher().reference_identifiers


def findings(response) -> list[dict]:
    """The structured findings from a rejected write."""
    return response.json()["error"]["detail"]["findings"]


class TestTheSeededGlobalUniverse:
    """A loaded dataset gets one universe for free."""

    def test_it_exists_on_a_fresh_workspace(self,
                                            client,
                                            known):
        body = client.get("/universes", headers=HEADERS).json()
        seeded = [u for u in body["universes"] if u["id"] == GLOBAL_ID]

        assert seeded, "a loaded dataset produced no universe at all"
        assert len(seeded[0]["identifiers"]) == len(known)

    def test_it_is_marked_seeded(self,
                                 client):
        """The client renders it read-only, so this has to be on the document
        rather than inferred from the id."""
        universe = client.get(f"/universes/{GLOBAL_ID}", headers=HEADERS).json()

        assert universe["source"] == "seeded"

    def test_it_holds_no_currency_pairs(self,
                                        client):
        """The reason the seed reads reference data rather than market data:
        an FX pair is a row set, not an instrument anybody can hold."""
        universe = client.get(f"/universes/{GLOBAL_ID}", headers=HEADERS).json()
        market_only = (set(dataset.data_fetcher().identifiers)
                       - set(dataset.data_fetcher().reference_identifiers))

        assert market_only, "the fixture has no FX pairs, so this proves nothing"
        assert not market_only & set(universe["identifiers"])

    def test_its_members_are_deterministic(self,
                                           known):
        """Sorted, so regenerating with the same seed reproduces the same
        document byte for byte rather than in whatever order the frame
        happened to yield."""
        from beacon.server.routers.universes import seed_global_universe
        from beacon.server.store import DocumentStore

        written = []
        for _ in range(2):
            store = DocumentStore("universes", root=Path(tempfile.mkdtemp()))
            seed_global_universe(store, dataset.data_fetcher())
            written.append(store.read(GLOBAL_ID))

        assert written[0] == written[1]
        assert written[0]["identifiers"] == sorted(known)

    def test_seeding_twice_writes_once(self):
        """Idempotent, so the file does not churn on every boot."""
        from beacon.server.routers.universes import seed_global_universe
        from beacon.server.store import DocumentStore

        store = DocumentStore("universes", root=Path(tempfile.mkdtemp()))

        assert seed_global_universe(store, dataset.data_fetcher()) is True
        assert seed_global_universe(store, dataset.data_fetcher()) is False

    def test_it_cannot_be_edited(self,
                                 client,
                                 known):
        """It derives from the dataset, so a regeneration would discard an
        edit. Refusing now beats losing it later."""
        response = client.put(f"/universes/{GLOBAL_ID}", headers=HEADERS,
                              json={"name": "Mine", "identifiers": known[:1]})

        assert response.status_code == 422
        assert "read-only" in response.json()["error"]["message"]

    def test_it_cannot_be_deleted(self,
                                  client):
        assert client.delete(f"/universes/{GLOBAL_ID}",
                             headers=HEADERS).status_code == 422

    def test_the_refusal_says_how_to_proceed(self,
                                             client,
                                             known):
        """"Read-only" without a way forward is a dead end; the message points
        at copying it."""
        response = client.delete(f"/universes/{GLOBAL_ID}", headers=HEADERS)

        assert "POST /universes" in response.json()["error"]["message"]


class TestCreatingAUniverse:
    """`POST /universes`, which did not exist."""

    def test_it_creates_and_is_immediately_readable(self,
                                                    client,
                                                    known):
        """The round-trip the client needs: create, list, read members."""
        response = client.post("/universes", headers=HEADERS,
                               json={"name": "My Three",
                                     "identifiers": known[:3]})

        assert response.status_code == 201

        created = response.json()

        assert created["id"] == "my-three"
        assert created["source"] == "user"

        listed = {u["id"] for u in
                  client.get("/universes", headers=HEADERS).json()["universes"]}

        assert "my-three" in listed

        members = client.get("/universes/my-three/members",
                             headers=HEADERS).json()

        assert members["identifiers"] == known[:3]

    def test_the_server_assigns_the_id(self,
                                       client,
                                       known):
        """A client cannot create two universes whose ids differ only in
        punctuation and expect them to be distinct documents."""
        response = client.post("/universes", headers=HEADERS,
                               json={"name": "  Tech & Media!  ",
                                     "identifiers": known[:1]})

        assert response.json()["id"] == "tech-media"

    @pytest.mark.parametrize("name,expected", [
        ("Simple", "simple"),
        ("Two  Words", "two-words"),
        ("UPPER_case", "upper-case"),
        ("trailing---", "trailing"),
    ])
    def test_the_slug_rules(self,
                            name,
                            expected):
        assert slug(name) == expected

    def test_a_name_with_no_letters_is_refused(self,
                                               client,
                                               known):
        """"!!!" has no identifier, and inventing one would produce a URL
        nobody could guess from the name."""
        response = client.post("/universes", headers=HEADERS,
                               json={"name": "!!!", "identifiers": known[:1]})

        assert response.status_code == 422
        assert findings(response)[0]["code"] == "UNUSABLE_NAME"

    def test_a_duplicate_name_is_refused(self,
                                         client,
                                         known):
        payload = {"name": "Repeat", "identifiers": known[:1]}

        assert client.post("/universes", headers=HEADERS,
                           json=payload).status_code == 201

        response = client.post("/universes", headers=HEADERS, json=payload)

        assert response.status_code == 422
        assert "already exists" in response.json()["error"]["message"]

    def test_duplicates_are_removed_silently(self,
                                             client,
                                             known):
        """Sending the same ticker twice is a paste artefact, not an error
        worth stopping for."""
        response = client.post("/universes", headers=HEADERS,
                               json={"name": "Duped",
                                     "identifiers": [known[0], known[0],
                                                     known[1]]})

        assert response.json()["identifiers"] == [known[0], known[1]]

    def test_the_member_order_is_kept(self,
                                      client,
                                      known):
        """A curated list has an order somebody chose; sorting it would
        discard that."""
        reversed_members = list(reversed(known[:3]))

        response = client.post("/universes", headers=HEADERS,
                               json={"name": "Ordered",
                                     "identifiers": reversed_members})

        assert response.json()["identifiers"] == reversed_members


class TestMembersAreCheckedAgainstTheData:
    """Both write paths, not just the new one."""

    def test_an_unknown_identifier_is_named(self,
                                            client,
                                            known):
        """The point of using findings rather than a bare 422: the response
        says *which* ticker was wrong."""
        response = client.post("/universes", headers=HEADERS,
                               json={"name": "Bad",
                                     "identifiers": [known[0], "NOSUCH"]})

        assert response.status_code == 422

        reported = findings(response)

        assert any(f["code"] == "UNKNOWN_IDENTIFIER" for f in reported)
        assert any("NOSUCH" in f["message"] for f in reported)

    def test_every_unknown_identifier_is_reported(self,
                                                  client,
                                                  known):
        """Fixing one typo only to be told about the next is a bad way to
        correct a list of five hundred."""
        response = client.post("/universes", headers=HEADERS,
                               json={"name": "Several",
                                     "identifiers": [known[0], "NOPE1",
                                                     "NOPE2", "NOPE3"]})

        messages = " ".join(f["message"] for f in findings(response))

        for missing in ("NOPE1", "NOPE2", "NOPE3"):
            assert missing in messages

    def test_a_long_list_of_typos_is_summarised(self,
                                                client):
        """A thousand bad tickers should not produce a thousand findings."""
        response = client.post("/universes", headers=HEADERS,
                               json={"name": "Many",
                                     "identifiers": [f"BAD{n}"
                                                     for n in range(200)]})

        reported = findings(response)

        assert len(reported) < 30
        assert any("further identifier" in f["message"] for f in reported)

    def test_an_empty_list_is_refused(self,
                                      client):
        response = client.post("/universes", headers=HEADERS,
                               json={"name": "Empty", "identifiers": []})

        assert response.status_code == 422
        assert findings(response)[0]["code"] == "EMPTY_UNIVERSE"

    def test_put_validates_its_members_too(self,
                                           client,
                                           known):
        """PUT predates the loaded data and accepted any list at all, so a
        universe could be edited into naming instruments with no prices — and
        the index built from it came back empty with nothing to point at."""
        response = client.put("/universes/edited", headers=HEADERS,
                              json={"name": "Edited",
                                    "identifiers": [known[0], "GHOST"]})

        assert response.status_code == 422
        assert any("GHOST" in f["message"] for f in findings(response))

    def test_put_still_accepts_a_valid_list(self,
                                            client,
                                            known):
        """Guards the test above: validation that refused everything would
        pass it."""
        response = client.put("/universes/edited", headers=HEADERS,
                              json={"name": "Edited",
                                    "identifiers": known[:2]})

        assert response.status_code == 200
        assert response.json()["identifiers"] == known[:2]


class TestWithoutADataSource:
    """A server started with no data can still hold universes."""

    def test_members_are_accepted_unchecked(self):
        """It cannot say whether they exist, and refusing everything would
        make the endpoint useless rather than safe."""
        app = create_app(ServerConfig(auth_token=TOKEN,
                                      storage_root=Path(tempfile.mkdtemp())))
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post("/universes", headers=HEADERS,
                               json={"name": "Blind",
                                     "identifiers": ["ANYTHING"]})

        assert response.status_code == 201

    def test_nothing_is_seeded(self):
        """There is no dataset to derive a GLOBAL universe from."""
        app = create_app(ServerConfig(auth_token=TOKEN,
                                      storage_root=Path(tempfile.mkdtemp())))
        client = TestClient(app, raise_server_exceptions=False)

        body = client.get("/universes", headers=HEADERS).json()

        assert body["universes"] == []


class TestTheSurface:
    """What the client generates from."""

    def test_post_is_in_the_spec(self,
                                 client):
        spec = client.get("/openapi.json").json()

        assert "post" in spec["paths"]["/universes"]

    def test_the_source_field_is_published(self,
                                           client):
        """beacon-ui branches on it to render a seeded universe read-only."""
        spec = client.get("/openapi.json").json()
        universe = spec["components"]["schemas"]["Universe"]["properties"]

        assert "source" in universe
