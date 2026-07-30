# src/beacon/tokens/__init__.py
"""
Design tokens: the colours Beacon draws with.

`colors.json` here is a **vendored copy** of `tokens/colors.json` in the
beacon-ui repository, which is generated from Figma and is the source of truth.
It is copied rather than fetched so this package installs and renders offline,
and a CI job fails when the two copies drift apart. See
`docs/decisions/0001-design-token-source-of-truth.md` for why.

Do not edit the values here. A change made in this file and not in beacon-ui is
a change the drift check will reject, and correctly so — the design system does
not live in this repository.

Two kinds of colour live in the file, and the distinction is deliberate:

* **tokens** carry a value per mode, light and dark. They are chrome: the
  colours a chart takes on so it sits inside the surrounding application.
* **raw** colours have one value regardless of mode. A correlation heatmap is a
  measurement scale rather than themed furniture and must not flip with the
  theme, and everything inside a report page is print ink that has to match the
  PDF it becomes.
"""
import json
import re
from functools import lru_cache
from importlib import resources
from typing import Any

from ..exceptions import ConfigurationError

LIGHT = "light"
DARK = "dark"
MODES = (LIGHT, DARK)

# Where the vendored copy lives, and the path it mirrors in beacon-ui. The
# drift check reads both, so they are named once, here.
FILENAME = "colors.json"
UPSTREAM_REPOSITORY = "karanbh01/beacon-ui"
UPSTREAM_PATH = "tokens/colors.json"

# A token whose source is anything but this was not read from a verified Figma
# variable — it is a placeholder or a value invented here and not yet mirrored
# back. Charts can still use it; callers who care can ask.
VERIFIED_SOURCE = "figma"

# #rgb, #rrggbb, or #rrggbbaa. Four tokens are deliberately translucent, so the
# 8-digit form is expected rather than a mistake.
HEX_COLOUR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    """Read and validate the token document.

    Read through importlib.resources rather than by path, so it works the same
    from a wheel, a zip import or a source checkout — and explicitly as UTF-8,
    because the file carries em dashes and the default encoding is not UTF-8 on
    every platform this runs on.

    Returns:
        dict: The parsed document. Cached; callers must not mutate it.

    Raises:
        ConfigurationError: If the file is missing, unparseable, or does not
            hold what the rest of this module promises.
    """
    try:
        source = resources.files(__package__).joinpath(FILENAME)
        document: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise ConfigurationError(
            "design tokens",
            f"{FILENAME} is missing from the installed package.") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            "design tokens", f"{FILENAME} is not valid JSON: {exc}.") from exc

    _validate(document)

    return document


def _validate(document: dict[str, Any]) -> None:
    """Reject a token file that would produce broken charts.

    Cheap, and it turns a malformed vendored copy into one clear error at load
    rather than a matplotlib failure deep inside a plotting call.
    """
    for section in ("modes", "tokens", "raw"):
        if section not in document:
            raise ConfigurationError("design tokens",
                                     f"{FILENAME} has no '{section}' section.")

    if list(document["modes"]) != list(MODES):
        raise ConfigurationError(
            "design tokens",
            f"{FILENAME} declares modes {document['modes']}, but this package "
            f"expects {list(MODES)}.")

    for name, token in document["tokens"].items():
        _validate_token(name, token)

    for name, value in document["raw"].items():
        if name != "comment" and not HEX_COLOUR.match(str(value)):
            raise ConfigurationError("design tokens",
                                     f"raw colour '{name}' is not a hex colour: {value!r}.")


def _validate_token(name: str,
                    token: dict[str, Any]) -> None:
    """Check one token carries a usable colour in every mode."""
    for mode in MODES:
        if mode not in token:
            raise ConfigurationError("design tokens",
                                     f"token '{name}' has no '{mode}' value.")

        if not HEX_COLOUR.match(str(token[mode])):
            raise ConfigurationError(
                "design tokens",
                f"token '{name}' has a {mode} value that is not a hex colour: "
                f"{token[mode]!r}.")


def token_names() -> list[str]:
    """Every token name, alphabetically."""
    return sorted(load()["tokens"])


def palette(mode: str = LIGHT) -> dict[str, str]:
    """Every token's colour in one mode.

    Args:
        mode: LIGHT or DARK.

    Returns:
        dict: Token name to hex colour. Values may carry an alpha channel;
        matplotlib accepts the 8-digit form directly.

    Raises:
        ConfigurationError: If *mode* is not a declared mode.
    """
    _require_mode(mode)

    return {name: token[mode] for name, token in load()["tokens"].items()}


def colour(name: str,
           mode: str = LIGHT) -> str:
    """One token's colour.

    Args:
        name: Token name, e.g. ``"accent"``.
        mode: LIGHT or DARK.

    Returns:
        str: Hex colour.

    Raises:
        ConfigurationError: If the token or mode is unknown. Unknown names are
            an error rather than a fallback: silently substituting a default
            would produce a chart that looks fine and is wrong.
    """
    _require_mode(mode)
    tokens = load()["tokens"]

    if name not in tokens:
        raise ConfigurationError(
            "design tokens",
            f"unknown token '{name}'. Available: {', '.join(token_names())}.")

    return str(tokens[name][mode])


def raw_colours() -> dict[str, str]:
    """The mode-independent colours, without the explanatory comment.

    Heatmap stops and report-page ink. These do not change with the theme by
    design — a measurement scale that flipped with the surrounding chrome would
    make two screenshots of the same data disagree.
    """
    return {name: str(value) for name, value in load()["raw"].items()
            if name != "comment"}


def unverified(mode: str = LIGHT) -> list[str]:
    """Tokens whose value in *mode* did not come from a verified Figma variable.

    A placeholder renders exactly as convincingly as a real colour, so the file
    records where each value came from and this surfaces it. Empty is the
    healthy state.

    Args:
        mode: LIGHT or DARK.

    Returns:
        list: Token names, alphabetically.
    """
    _require_mode(mode)

    return sorted(name for name, token in load()["tokens"].items()
                  if token.get("source", {}).get(mode) != VERIFIED_SOURCE)


def _require_mode(mode: str) -> None:
    """Raise unless *mode* is one this file defines."""
    if mode not in MODES:
        raise ConfigurationError(
            "design tokens",
            f"unknown mode '{mode}'. Available: {', '.join(MODES)}.")
