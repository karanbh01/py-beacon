# tests/test_tokens.py
"""BN-94: the vendored design tokens and their loader."""
import json
from unittest.mock import patch

import pytest

from beacon import tokens
from beacon.exceptions import ConfigurationError

# Values read straight from beacon-ui's Figma export. Hard-coded rather than
# read back from the same file the loader reads, so this test would catch a
# vendored copy that was silently replaced as well as a loader that misreads it.
ACCENT_LIGHT = "#4a88c7"
ACCENT_DARK = "#6fa7dc"

# Four tokens are translucent by design; surface is one, and its 8-digit hex is
# a value matplotlib accepts rather than a malformed colour.
TRANSLUCENT_SURFACE_LIGHT = "#fbf3e233"


@pytest.fixture(autouse=True)
def _clear_cache():
    """The loader caches; each test starts from a clean read."""
    tokens.load.cache_clear()
    yield
    tokens.load.cache_clear()


class TestLoading:

    def test_the_document_loads(self):
        document = tokens.load()

        assert set(document) >= {"modes", "tokens", "raw"}

    def test_it_declares_both_modes(self):
        assert tokens.load()["modes"] == ["light", "dark"]
        assert tokens.MODES == ("light", "dark")

    def test_every_token_carries_both_modes(self):
        for name, token in tokens.load()["tokens"].items():
            assert "light" in token, f"{name} has no light value"
            assert "dark" in token, f"{name} has no dark value"

    def test_it_is_read_as_utf8_regardless_of_platform(self):
        """The file carries em dashes and the default encoding is not UTF-8
        everywhere. Reading it with the platform default mangles the prose, and
        on a comparison against upstream that reads as drift where there is
        none — which is exactly how this was found.
        """
        descriptions = " ".join(token["description"]
                                for token in tokens.load()["tokens"].values())

        assert "—" in descriptions
        assert "�" not in descriptions

    def test_the_result_is_cached(self):
        assert tokens.load() is tokens.load()


class TestPalette:

    def test_it_covers_every_token(self):
        assert set(tokens.palette()) == set(tokens.token_names())

    def test_the_two_modes_differ(self):
        """A palette that came back identical would mean mode is being ignored."""
        assert tokens.palette(tokens.LIGHT) != tokens.palette(tokens.DARK)

    def test_light_is_the_default(self):
        assert tokens.palette() == tokens.palette(tokens.LIGHT)

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ConfigurationError, match="unknown mode"):
            tokens.palette("sepia")


class TestColour:

    def test_it_returns_the_figma_value(self):
        assert tokens.colour("accent") == ACCENT_LIGHT
        assert tokens.colour("accent", tokens.DARK) == ACCENT_DARK

    def test_translucent_tokens_keep_their_alpha(self):
        """Dropping the alpha would make a wash render as an opaque fill."""
        assert tokens.colour("surface") == TRANSLUCENT_SURFACE_LIGHT
        assert len(tokens.colour("surface")) == 9

    def test_every_colour_is_a_hex_string(self):
        for mode in tokens.MODES:
            for name, value in tokens.palette(mode).items():
                assert tokens.HEX_COLOUR.match(value), f"{name}/{mode} is {value!r}"

    def test_an_unknown_token_is_refused(self):
        """Substituting a default would give a chart that looks fine and is wrong."""
        with pytest.raises(ConfigurationError, match="unknown token"):
            tokens.colour("not-a-token")

    def test_the_error_lists_what_is_available(self):
        with pytest.raises(ConfigurationError, match="accent"):
            tokens.colour("not-a-token")


class TestRawColours:

    def test_the_comment_is_not_a_colour(self):
        assert "comment" not in tokens.raw_colours()

    def test_it_carries_the_heatmap_stops(self):
        """BN-81's correlation colormap is built from these."""
        raw = tokens.raw_colours()

        assert {"heatmap-low", "heatmap-mid", "heatmap-high"} <= set(raw)

    def test_it_carries_the_report_page_ink(self):
        """BN-97's PDF output is built from these."""
        raw = tokens.raw_colours()

        assert {"paper-page", "paper-ink", "paper-rule"} <= set(raw)

    def test_every_raw_value_is_a_hex_string(self):
        for name, value in tokens.raw_colours().items():
            assert tokens.HEX_COLOUR.match(value), f"{name} is {value!r}"

    def test_raw_colours_do_not_vary_by_mode(self):
        """Each raw entry is one colour, not a light/dark pair.

        A measurement scale that flipped with the theme would make two
        screenshots of the same data disagree, which is the whole reason these
        sit outside the mode system.
        """
        for name, value in tokens.raw_colours().items():
            assert isinstance(value, str), f"{name} is not a single value"
            assert tokens.LIGHT not in str(value)
            assert tokens.DARK not in str(value)

    def test_no_raw_name_collides_with_a_token_name(self):
        """The two sets are addressed differently, so an overlap would be a trap."""
        assert not set(tokens.raw_colours()) & set(tokens.token_names())


class TestProvenance:

    def test_unverified_tokens_are_surfaced(self):
        """A placeholder renders as convincingly as a real colour."""
        unverified = tokens.unverified()

        assert "series-2" in unverified
        assert "series-3" in unverified

    def test_verified_tokens_are_not_listed(self):
        assert "accent" not in tokens.unverified()
        assert "accent" not in tokens.unverified(tokens.DARK)

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ConfigurationError, match="unknown mode"):
            tokens.unverified("sepia")


class TestValidation:
    """A malformed vendored copy must fail at load with one clear error.

    Every case here patches the parsed document rather than the file, so the
    real tokens are never touched.
    """

    def _load_with(self,
                   document):
        tokens.load.cache_clear()

        with patch.object(tokens.json, "loads", return_value=document):
            return tokens.load()

    def test_a_missing_section_is_refused(self):
        with pytest.raises(ConfigurationError, match="no 'tokens' section"):
            self._load_with({"modes": ["light", "dark"], "raw": {}})

    def test_unexpected_modes_are_refused(self):
        with pytest.raises(ConfigurationError, match="declares modes"):
            self._load_with({"modes": ["light"], "tokens": {}, "raw": {}})

    def test_a_token_missing_a_mode_is_refused(self):
        document = {"modes": ["light", "dark"],
                    "tokens": {"accent": {"light": "#ffffff"}},
                    "raw": {}}

        with pytest.raises(ConfigurationError, match="has no 'dark' value"):
            self._load_with(document)

    def test_a_token_that_is_not_a_colour_is_refused(self):
        document = {"modes": ["light", "dark"],
                    "tokens": {"accent": {"light": "cornflower", "dark": "#000000"}},
                    "raw": {}}

        with pytest.raises(ConfigurationError, match="not a hex colour"):
            self._load_with(document)

    def test_a_raw_value_that_is_not_a_colour_is_refused(self):
        document = {"modes": ["light", "dark"],
                    "tokens": {},
                    "raw": {"heatmap-low": "greenish"}}

        with pytest.raises(ConfigurationError, match="not a hex colour"):
            self._load_with(document)

    def test_the_raw_comment_is_exempt_from_colour_validation(self):
        document = {"modes": ["light", "dark"],
                    "tokens": {},
                    "raw": {"comment": ["prose, not a colour"]}}

        assert self._load_with(document)["raw"]["comment"]

    def test_a_missing_file_is_refused(self):
        """A build that stopped shipping the JSON must say so, not crash later."""
        tokens.load.cache_clear()

        with (patch.object(tokens.resources, "files",
                           side_effect=FileNotFoundError("gone")),
              pytest.raises(ConfigurationError, match="missing from the installed")):
            tokens.load()

    def test_unparseable_json_is_refused(self):
        tokens.load.cache_clear()

        with (patch.object(tokens.json, "loads",
                           side_effect=json.JSONDecodeError("bad", "", 0)),
              pytest.raises(ConfigurationError, match="not valid JSON")):
            tokens.load()


class TestPackaging:
    """The tokens are data, so nothing else notices if they stop shipping."""

    def test_the_file_sits_inside_the_package(self):
        """At the repository root it would not ship in the wheel, and BN-77's
        style generation would fail for anyone who pip-installed rather than
        cloned.
        """
        from importlib import resources

        assert resources.files("beacon.tokens").joinpath("colors.json").is_file()

    def test_it_loads_without_any_optional_dependency(self):
        """Standard library only, so the tokens are core rather than behind an
        extra — a chart backend needs them before it needs matplotlib.
        """
        import subprocess
        import sys

        from beacon._optional import EXTRA_FOR_MODULE

        script = (
            "import sys\n"
            f"blocked = {sorted(EXTRA_FOR_MODULE)!r}\n"
            "class Blocker:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in blocked:\n"
            "            raise ImportError(name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "from beacon import tokens\n"
            "assert tokens.colour('accent')\n"
            "print('ok')\n"
        )

        completed = subprocess.run([sys.executable, "-c", script],
                                   capture_output=True, text=True, check=False)

        assert completed.returncode == 0, completed.stderr
        assert "ok" in completed.stdout


def _drift_module():
    """Load the drift-check script by path.

    It lives in scripts/ rather than in the package because it is CI tooling,
    not something a user of the library should be shipped. That keeps it off
    sys.path, so it is loaded explicitly here rather than imported.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "check_token_drift.py"
    spec = importlib.util.spec_from_file_location("check_token_drift", path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def _document(accent_light: str = "#4a88c7",
              heatmap_low: str = "#4ca56b",
              description: str = "Primary chart series.") -> dict:
    """A minimal token document in the real file's shape."""
    return {
        "modes": ["light", "dark"],
        "tokens": {
            "accent": {"description": description,
                       "light": accent_light,
                       "dark": "#6fa7dc",
                       "source": {"light": "figma", "dark": "figma"}},
        },
        "raw": {"comment": ["prose"], "heatmap-low": heatmap_low},
    }


class TestDriftDetection:
    """The check has to notice a real change and ignore a cosmetic one."""

    @pytest.fixture(scope="class")
    def drift(self):
        return _drift_module()

    def test_identical_documents_agree(self,
                                       drift):
        assert drift.differences(drift.comparable(_document()),
                                 drift.comparable(_document())) == []

    def test_a_changed_colour_is_caught(self,
                                        drift):
        found = drift.differences(drift.comparable(_document()),
                                  drift.comparable(_document(accent_light="#000000")))

        assert len(found) == 1
        assert "accent" in found[0]

    def test_a_changed_raw_colour_is_caught(self,
                                            drift):
        found = drift.differences(drift.comparable(_document()),
                                  drift.comparable(_document(heatmap_low="#000000")))

        assert len(found) == 1
        assert "heatmap-low" in found[0]

    def test_reworded_prose_is_not_drift(self,
                                         drift):
        """Rewording a description is not a design change and must not fail a build."""
        found = drift.differences(
            drift.comparable(_document()),
            drift.comparable(_document(description="Totally rewritten prose.")))

        assert found == []

    def test_a_token_added_upstream_is_caught(self,
                                              drift):
        upstream = _document()
        upstream["tokens"]["series-4"] = {"light": "#111111", "dark": "#222222",
                                          "source": {"light": "new", "dark": "new"}}

        found = drift.differences(drift.comparable(_document()),
                                  drift.comparable(upstream))

        assert any("added upstream" in line for line in found)

    def test_a_token_removed_upstream_is_caught(self,
                                                drift):
        vendored = _document()
        vendored["tokens"]["legacy"] = {"light": "#111111", "dark": "#222222",
                                        "source": {"light": "figma", "dark": "figma"}}

        found = drift.differences(drift.comparable(vendored),
                                  drift.comparable(_document()))

        assert any("removed upstream" in line for line in found)

    def test_changed_modes_are_caught(self,
                                      drift):
        upstream = _document()
        upstream["modes"] = ["light", "dark", "high-contrast"]

        found = drift.differences(drift.comparable(_document()),
                                  drift.comparable(upstream))

        assert any("modes" in line for line in found)

    def test_an_unreachable_upstream_is_not_a_pass(self,
                                                   drift,
                                                   monkeypatch,
                                                   capsys):
        """A check that could not run has not run."""
        import urllib.error

        def refuse(*args, **kwargs):
            raise urllib.error.URLError("no network")

        monkeypatch.setattr(drift.urllib.request, "urlopen", refuse)

        assert drift.main() == drift.UNREACHABLE
        assert "NOT verified" in capsys.readouterr().out

    def test_the_vendored_copy_is_what_the_package_loads(self,
                                                          drift):
        """The check and the loader must read the same file, or it checks nothing."""
        assert drift.VENDORED.name == tokens.FILENAME
        assert drift.VENDORED.read_text(encoding="utf-8")
        assert drift.REPOSITORY == tokens.UPSTREAM_REPOSITORY
