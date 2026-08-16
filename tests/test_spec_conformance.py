# tests/test_spec_conformance.py
"""BN-131: the endpoint defects the first working fuzz run found.

The nightly schemathesis job had never executed a single case (BN-124), so
none of this had been looked at. Its first real sweep produced 35 unique
failures, eleven of them 5xx.

These are unit tests rather than a second fuzzer. The fuzz run finds *which*
inputs break; a test pins the behaviour so it cannot regress, and says why the
answer is what it is. The nightly keeps looking for new ones.

**The theme is whose fault an error is.** Every 5xx here was the server
correctly detecting bad input and then reporting itself as broken -- the
path-traversal guard, the date validators, the block dispatcher. A 500 tells a
client to retry and page someone; a 422 tells it to fix the request. Getting
that wrong is not cosmetic.
"""
import logging
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from beacon.exceptions import InvalidIdentifierError
from beacon.server import ServerConfig, create_app
from beacon.server.schemas import IDENTIFIER_PATTERN, ISO_DATE_PATTERN
from beacon.testing import dataset

TOKEN = "spec-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def client():
    """A configured server over the canonical fixture."""
    root = tempfile.mkdtemp()

    app = create_app(ServerConfig(auth_token=TOKEN,
                                  data_fetcher=dataset.data_fetcher(),
                                  storage_root=Path(root)))

    return TestClient(app, raise_server_exceptions=False)


def trs_body(date: str) -> dict:
    """A TRS pricing request with every date set to `date`."""
    return {"start_date": date, "end_date": date, "valuation_date": date,
            "notional": 1.0, "spot": 1.0, "initial_price": 1.0}


class TestAnInvalidIdentifierIsTheCallersFault:
    """The path-traversal guard, which accounted for most of the 5xx.

    It worked -- it detected `..`, refused it, and explained why. Then it
    raised `ConfigurationError`, which means *the server is misconfigured* and
    maps to 500. So a client probing for traversal was told it had found a
    server fault, and a client with a typo was told to page an operator.
    """

    @pytest.mark.parametrize("path", [
        "/universes/0..0",
        "/indices/0..0",
        "/data/watchlists/0..0",
        "/reports/templates/0..0",
        "/beacon/0..0/overview",
    ])
    def test_a_traversal_attempt_is_refused_not_a_server_error(self,
                                                               client,
                                                               path):
        response = client.get(path, headers=HEADERS)

        assert response.status_code == 422, (
            f"{path} answered {response.status_code}")

    @pytest.mark.parametrize("path", [
        "/universes/0..0",
        "/data/watchlists/0..0",
    ])
    def test_it_is_refused_on_delete_too(self,
                                         client,
                                         path):
        """A traversal on DELETE is the one that would have written outside
        the collection, so it matters more than the read."""
        assert client.delete(path, headers=HEADERS).status_code == 422

    def test_an_empty_identifier_is_refused(self,
                                            client):
        """`""` raised a bare `ValueError` with no handler, so it reached the
        client as a 500 carrying no code and no message at all."""
        response = client.post("/reports/render", headers=HEADERS,
                               json={"template_id": "", "index_id": None})

        assert response.status_code == 422

    def test_the_error_does_not_echo_the_whole_input(self):
        """The value came from a URL and may be long or hostile. A client
        needs enough to recognise which id it sent, not the entire string
        handed back to it."""
        hostile = "../" * 200

        error = InvalidIdentifierError(hostile, "it must not contain path "
                                                "separators.")

        assert len(error.identifier) < 60
        assert str(error).startswith("Invalid identifier")

    def test_a_valid_identifier_still_works(self,
                                            client):
        """The guard must not have been tightened into refusing real ids."""
        response = client.put("/data/watchlists/my-list_1", headers=HEADERS,
                              json={"name": "Mine", "identifiers": ["AAA"]})

        assert response.status_code in (200, 201)


class TestDatesAreCheckedAtTheEdge:
    """Date fields were plain strings, so anything reached the parser."""

    @pytest.mark.parametrize("value", ["", "0000-00-00", "2024-02-31",
                                       "not-a-date", "2024-1-5"])
    def test_an_unusable_date_is_refused(self,
                                         client,
                                         value):
        """Shape *and* calendar. `0000-00-00` matches `\\d{4}-\\d{2}-\\d{2}`
        and is still not a date anyone could observe, so the pattern alone
        left a 500 behind."""
        response = client.post("/derivatives/trs/price", headers=HEADERS,
                               json=trs_body(value))

        assert response.status_code == 422, f"{value!r} was accepted"

    def test_a_real_date_is_accepted(self,
                                     client):
        """Guards the tests above: a validator that refused everything would
        pass all of them."""
        response = client.post(
            "/derivatives/trs/price", headers=HEADERS,
            json={"start_date": "2024-01-15", "end_date": "2025-01-15",
                  "valuation_date": "2024-06-15", "notional": 1e6,
                  "spot": 100.0, "initial_price": 100.0})

        assert response.status_code == 200

    def test_the_pattern_is_in_the_spec(self,
                                        client):
        """The point of constraining at the schema rather than in code: a
        client can read the contract instead of discovering it.

        Navigated rather than string-matched. Searching `str(spec)` for the
        pattern fails even when it is present, because a dict repr escapes
        the backslashes -- which is how this test failed first time round.
        """
        spec = client.get("/openapi.json").json()

        dates = spec["components"]["schemas"]["TrsPriceRequest"]["properties"]

        assert dates["start_date"]["pattern"] == ISO_DATE_PATTERN

        parameters = spec["paths"]["/universes/{universe_id}"]["get"]["parameters"]
        identifiers = [parameter["schema"] for parameter in parameters
                       if parameter["name"] == "universe_id"]

        assert identifiers, "the path parameter is not in the spec"
        assert identifiers[0]["pattern"] == IDENTIFIER_PATTERN


class TestAValueErrorIsNotAServerFault:
    """The library validates with bare `ValueError` in many places, and none
    of them had a handler."""

    def test_a_rejected_argument_answers_422_with_its_reason(self,
                                                             client):
        """`end_date must be after start_date` is a true and useful sentence
        that the client never saw: the response was a bare 500 with no body
        beyond `Internal Server Error`."""
        response = client.post(
            "/derivatives/trs/price", headers=HEADERS,
            json={"start_date": "2024-01-15", "end_date": "2024-01-15",
                  "valuation_date": "2024-01-15", "notional": 1.0,
                  "spot": 1.0, "initial_price": 1.0})

        assert response.status_code == 422

        error = response.json()["error"]

        assert error["code"] == "INVALID_ARGUMENT"
        assert "after start_date" in error["message"]

    def test_it_is_logged_as_an_error(self,
                                      client,
                                      caplog):
        """The mitigation for mapping every ValueError to the caller's fault:
        a genuine internal one is still visible to whoever runs the server,
        rather than being quietly relabelled."""
        with caplog.at_level(logging.ERROR, logger="beacon.server.errors"):
            client.post("/derivatives/trs/price", headers=HEADERS,
                        json=trs_body("2024-01-15"))

        assert any("ValueError" in record.message
                   or "ValueError" in record.getMessage()
                   for record in caplog.records)


class TestAMalformedBlockIsRefusedOnSave:
    """`{"blocks": [{}]}` reached the renderer and came back as 500."""

    def test_an_unknown_block_kind_is_refused(self,
                                              client):
        response = client.put("/reports/templates/t1", headers=HEADERS,
                              json={"name": "T", "template_id": "t1",
                                    "blocks": [{}]})

        assert response.status_code == 422

    def test_the_error_names_the_position_and_the_options(self,
                                                          client):
        """A template can carry many blocks, so "one of them is wrong" is not
        an answer anybody can act on."""
        response = client.put("/reports/templates/t2", headers=HEADERS,
                              json={"name": "T", "template_id": "t2",
                                    "blocks": [{"kind": "text", "body": "ok"},
                                               {"kind": "nonsense"}]})

        assert response.status_code == 422

        rendered = str(response.json())

        assert "block 1" in rendered
        assert "text" in rendered

    def test_a_good_template_still_saves(self,
                                         client):
        response = client.put("/reports/templates/t3", headers=HEADERS,
                              json={"name": "T", "template_id": "t3",
                                    "blocks": [{"kind": "text",
                                                "body": "hello"}]})

        assert response.status_code == 200


class TestAnUnreadableBodyIsRefused:
    """Found while chasing the undocumented 400, not by the fuzzer.

    schemathesis always sends a correct `Content-Type`, so it never tried
    these -- a reminder that a fuzzer explores the space its generator knows
    about, and a wrong header is one of the commonest things a real client
    does.
    """

    @pytest.mark.parametrize("headers,body", [
        ({"Content-Type": "text/plain"}, b"x"),
        ({}, b"{}"),
        ({"Content-Type": "application/json"}, b"{not json"),
    ])
    def test_it_answers_422_rather_than_500(self,
                                            client,
                                            headers,
                                            body):
        response = client.post("/indices/validate",
                               headers={**HEADERS, **headers}, content=body)

        assert response.status_code == 422

    def test_the_response_body_survives_serialisation(self,
                                                      client):
        """The failure was a `TypeError` while encoding the error, so a status
        code alone would not have caught it: pydantic puts the raw request
        **bytes** into the error's `input`, and `JSONResponse` cannot encode
        them."""
        import json

        response = client.post("/indices/validate",
                               headers={**HEADERS, "Content-Type": "text/plain"},
                               content=b"plain text body")

        json.dumps(response.json())

        assert response.json()["error"]["code"] == "VALIDATION_ERROR"


class TestEveryStatusItReturnsIsDocumented:
    """A client generated from the spec has no branch for a status the spec
    does not mention, and meets it first in production."""

    @pytest.mark.parametrize("status", [400, 401, 404, 405, 422, 500, 501, 503])
    def test_the_error_statuses_are_declared(self,
                                             client,
                                             status):
        spec = client.get("/openapi.json").json()
        documented = spec["paths"]["/indices/validate"]["post"]["responses"]

        assert str(status) in documented

    def test_a_wrong_method_answers_405(self,
                                        client):
        response = client.patch("/indices/validate", headers=HEADERS, json={})

        assert response.status_code == 405
        assert response.json()["error"]["code"] == "METHOD_NOT_ALLOWED"


class TestReservedIdentifiers:
    """A path segment that names an endpoint cannot also name a document."""

    def test_reserved_identifiers_match_the_routes(self,
                                                   client):
        """Derived from the running app rather than trusted.

        A hand-kept list is a list that drifts: the next endpoint added beside
        a path parameter would reintroduce the collision silently. This walks
        the real routes, finds every literal segment that sits where a path
        parameter also sits, and checks the constant covers it.
        """
        from beacon.server.store import RESERVED_IDENTIFIERS

        paths = client.get("/openapi.json").json()["paths"]

        # Prefixes that have a `{...}` child, and the literal children of the
        # same prefix. A literal at that position is exactly the ambiguity.
        parameterised: set[str] = set()
        literals: dict[str, set[str]] = {}

        for path in paths:
            head, _, tail = path.rpartition("/")

            if tail.startswith("{"):
                parameterised.add(head)
            elif tail and not tail.startswith("{"):
                literals.setdefault(head, set()).add(tail)

        colliding = {segment
                     for prefix in parameterised
                     for segment in literals.get(prefix, set())}

        assert colliding <= RESERVED_IDENTIFIERS, (
            f"these path literals collide with a document id and are not "
            f"reserved: {sorted(colliding - RESERVED_IDENTIFIERS)}")

    @pytest.mark.parametrize("reserved", ["preview", "validate", "rule-types"])
    def test_a_reserved_name_cannot_be_saved(self,
                                             client,
                                             reserved):
        """`PUT /optimise/constraint-sets/validate` returned 200 before this,
        having stored a constraint set whose id was literally `validate`."""
        response = client.put(f"/data/watchlists/{reserved}", headers=HEADERS,
                              json={"name": "x", "identifiers": ["AAA"]})

        assert response.status_code == 422

    def test_the_message_says_why(self,
                                  client):
        """"Invalid identifier" without a reason sends somebody to the source
        to find out which names are special."""
        response = client.put("/data/watchlists/preview", headers=HEADERS,
                              json={"name": "x", "identifiers": ["AAA"]})

        assert "reserved" in str(response.json()).lower()

    def test_ordinary_names_are_unaffected(self,
                                           client):
        """The reservation is three words, not a new naming policy."""
        for identifier in ("previews", "validated", "my-index", "rule_types"):
            response = client.put(f"/data/watchlists/{identifier}",
                                  headers=HEADERS,
                                  json={"name": "x", "identifiers": ["AAA"]})

            assert response.status_code in (200, 201), identifier


class TestTheFrequencyStaysAStringOnPurpose:
    """Declaring the four cadences would silence a fuzz finding and cost the
    thing the finding was protecting.

    `POST /indices` with an unknown cadence is refused *with a coded finding*
    -- `UNSUPPORTED_FREQUENCY`, in `error.detail.findings` -- which beacon-ui
    renders against the field. A `Literal` moves the refusal into pydantic,
    which answers with a generic validation error and no code.

    So schemathesis's "API rejected schema-compliant request" stands here by
    design. This test records that, so the next person to notice the loose
    type finds the reason rather than the opportunity.
    """

    def test_the_library_still_owns_the_valid_set(self):
        """The four cadences live in `beacon.index.schedule`, and
        `definitions.py` checks against them. The refusal itself is covered by
        `test_server_indices.py::test_unsupported_frequency_is_reported`,
        which has the fixtures to build a document that gets that far -- this
        only pins where the list lives."""
        from beacon.index.schedule import FREQUENCIES

        assert set(FREQUENCIES) == {"MONTHLY", "QUARTERLY",
                                    "SEMI-ANNUAL", "ANNUAL"}

    def test_the_field_is_not_an_enum_in_the_spec(self,
                                                  client):
        """Guards the decision itself: making it an enum would pass every
        other test in this file while quietly dropping the code above."""
        spec = client.get("/openapi.json").json()
        field = spec["components"]["schemas"]["IndexDocument"]["properties"]

        assert "enum" not in field["rebalancing_frequency"]


class TestValidationErrorsSerialise:
    """The bug that hid the block check: the handler for validation errors
    could not serialise its own output."""

    def test_a_custom_validator_reaches_the_client(self,
                                                   client):
        """pydantic v2 puts the raised `ValueError` *object* in the error's
        ``ctx``, and `JSONResponse` cannot encode it -- so the handler for
        422s raised `TypeError` and the client got a 500 instead. Every custom
        field validator was a latent 500, which went unnoticed because no
        schema had one until this issue added the block-kind check.
        """
        response = client.put("/reports/templates/t4", headers=HEADERS,
                              json={"name": "T", "template_id": "t4",
                                    "blocks": [{"kind": "?"}]})

        assert response.status_code == 422

        body = response.json()

        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert body["error"]["detail"]["errors"], "the errors were dropped"

    def test_the_response_is_valid_json_throughout(self,
                                                   client):
        """The failure mode was a serialisation error, so the assertion has to
        be that the whole body survives -- not merely that a status arrived."""
        import json

        response = client.put("/reports/templates/t5", headers=HEADERS,
                              json={"name": "T", "template_id": "t5",
                                    "blocks": [{"kind": None}]})

        json.dumps(response.json())
