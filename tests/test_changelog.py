# tests/test_changelog.py
"""BN-222: the engine serves its own changelog.

beacon-ui shows the user what the running engine has changed, so the notes
come from the engine, like its version does. These tests cover the parser,
the "what's new since" filter, the endpoint, and two checks on the file
itself: every release has its section, and it stays in the house style.
"""
import pytest
from fastapi.testclient import TestClient

from beacon import __version__
from beacon.changelog import entries, newer_than, parse, read
from beacon.server import ServerConfig, create_app

SAMPLE = """# Changelog

Intro prose, which belongs to no release.

## [Unreleased]

## [0.10.0] - 2027-01-05

### Added

- A thing with `code`.
- A long item that
  wraps onto a second line.

### Fixed

- A bug.

## [0.9.0] - 2026-12-01

Prose under a release but before a section is skipped.

### Changed

- Something else.

[Unreleased]: https://example.com/compare
[0.10.0]: https://example.com/0.10.0
"""


class TestParsing:

    def test_releases_come_newest_first_with_their_dates(self):
        parsed = parse(SAMPLE)

        assert [(entry.version, entry.date) for entry in parsed] == [
            ("Unreleased", None), ("0.10.0", "2027-01-05"), ("0.9.0", "2026-12-01")]

    def test_sections_hold_their_items_as_markdown(self):
        added = parse(SAMPLE)[1].sections[0]

        assert added.heading == "Added"
        assert added.items[0] == "A thing with `code`."

    def test_a_wrapped_item_is_one_item(self):
        assert parse(SAMPLE)[1].sections[0].items[1] == (
            "A long item that wraps onto a second line.")

    def test_prose_and_link_references_are_not_items(self):
        parsed = parse(SAMPLE)

        assert parsed[0].sections == []
        assert parsed[2].sections[0].items == ["Something else."]


class TestWhatsNew:

    def test_only_later_releases_are_returned(self):
        assert [entry.version for entry in newer_than(parse(SAMPLE), "0.9.0")] == [
            "Unreleased", "0.10.0"]

    def test_versions_compare_as_numbers_not_text(self):
        """As text, "0.10.0" sorts before "0.9.0"."""
        assert "0.10.0" in [entry.version
                            for entry in newer_than(parse(SAMPLE), "0.9.0")]

    def test_a_newer_version_than_any_listed_returns_no_releases(self):
        """An app that last saw a newer engine must not be shown the whole
        history as new."""
        released = [entry for entry in newer_than(parse(SAMPLE), "1.4.0")
                    if entry.version != "Unreleased"]

        assert released == []

    def test_a_malformed_version_is_refused(self):
        with pytest.raises(ValueError, match="not a version"):
            newer_than(parse(SAMPLE), "latest")


class TestTheFileItself:

    def test_the_running_version_has_a_dated_section(self):
        """The release workflow reads this section for the GitHub release
        notes, and the app shows it. A version bump without one is caught
        here rather than at tag time."""
        release = {entry.version: entry for entry in entries()}

        assert __version__ in release, (
            f"CHANGELOG.md has no section for {__version__}")
        assert release[__version__].date is not None

    def test_it_is_written_without_em_dashes(self):
        """House style for the changelog, which users read in the app."""
        assert "—" not in read()


class TestTheEndpoint:

    @pytest.fixture
    def client(self):
        return TestClient(create_app(ServerConfig(auth_token="t")))

    def test_it_serves_the_running_versions_notes(self,
                                                  client):
        body = client.get("/changelog",
                          headers={"Authorization": "Bearer t"}).json()

        assert body["version"] == __version__
        assert __version__ in [entry["version"] for entry in body["entries"]]

    def test_since_narrows_it_to_what_is_new(self,
                                             client):
        body = client.get("/changelog", params={"since": __version__},
                          headers={"Authorization": "Bearer t"}).json()

        assert __version__ not in [entry["version"] for entry in body["entries"]]

    def test_a_malformed_since_is_the_callers_fault(self,
                                                    client):
        response = client.get("/changelog", params={"since": "latest"},
                              headers={"Authorization": "Bearer t"})

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INVALID_ARGUMENT"

    def test_an_empty_release_is_left_out(self,
                                          client):
        """The Unreleased heading sits empty between releases."""
        body = client.get("/changelog",
                          headers={"Authorization": "Bearer t"}).json()

        assert all(entry["sections"] for entry in body["entries"])

    def test_it_needs_the_token_like_every_route(self,
                                                 client):
        assert client.get("/changelog").status_code == 401
