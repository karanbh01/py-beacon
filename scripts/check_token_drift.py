# scripts/check_token_drift.py
"""
Fail when the vendored design tokens drift from beacon-ui's copy.

The tokens are generated from Figma in beacon-ui and copied into this package
so that it installs and renders offline (see
docs/decisions/0001-design-token-source-of-truth.md). A copy is only safe if
divergence is loud, which is what this script is for.

The comparison is **semantic, not byte-for-byte**: both files are parsed and
their token values compared. What matters is that the two sides agree on the
colours, not that they agree on whitespace, key order or line endings — the
last of which would fail on a Windows checkout for no useful reason.

Run it by hand with:

    python scripts/check_token_drift.py

Exit codes: 0 agreement, 1 drift, 2 upstream unreachable. CI treats 2 as a
failure too, because a check that cannot run is not a check that passed —
but it is reported differently so a network blip is not mistaken for a
design-system change.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPOSITORY = "karanbh01/beacon-ui"
BRANCH = "main"
UPSTREAM_URL = f"https://raw.githubusercontent.com/{REPOSITORY}/{BRANCH}/tokens/colors.json"
VENDORED = Path(__file__).resolve().parent.parent / "src" / "beacon" / "tokens" / "colors.json"

TIMEOUT_SECONDS = 30

AGREE = 0
DRIFTED = 1
UNREACHABLE = 2

# Prose, not values. A reworded description is not a design change and should
# not fail anyone's build; a changed hex is and must.
IGNORED_KEYS = ("comment", "description")


def fetch_upstream() -> dict[str, Any]:
    """Read beacon-ui's copy of the tokens.

    Decoded explicitly as UTF-8: the file carries em dashes, and relying on the
    platform default would make this pass on one runner and fail on another.
    """
    with urllib.request.urlopen(UPSTREAM_URL, timeout=TIMEOUT_SECONDS) as response:
        payload: bytes = response.read()

    return dict(json.loads(payload.decode("utf-8")))


def read_vendored() -> dict[str, Any]:
    """Read this repository's copy."""
    return dict(json.loads(VENDORED.read_text(encoding="utf-8")))


def comparable(document: dict[str, Any]) -> dict[str, Any]:
    """Strip the parts that carry no colour information."""
    tokens = {
        name: {key: value for key, value in token.items() if key not in IGNORED_KEYS}
        for name, token in document.get("tokens", {}).items()
    }
    raw = {name: value for name, value in document.get("raw", {}).items()
           if name not in IGNORED_KEYS}

    return {"modes": document.get("modes"), "tokens": tokens, "raw": raw}


def differences(vendored: dict[str, Any],
                upstream: dict[str, Any]) -> list[str]:
    """Every disagreement between the two, as readable lines."""
    found: list[str] = []

    if vendored["modes"] != upstream["modes"]:
        found.append(f"modes: vendored {vendored['modes']}, "
                     f"upstream {upstream['modes']}")

    found.extend(_compare_section("token", vendored["tokens"], upstream["tokens"]))
    found.extend(_compare_section("raw", vendored["raw"], upstream["raw"]))

    return found


def _compare_section(label: str,
                     vendored: dict[str, Any],
                     upstream: dict[str, Any]) -> list[str]:
    """Compare one section, reporting additions, removals and changes."""
    added = [f"{label} '{name}': added upstream, missing here"
             for name in sorted(set(upstream) - set(vendored))]

    removed = [f"{label} '{name}': present here, removed upstream"
               for name in sorted(set(vendored) - set(upstream))]

    changed = [f"{label} '{name}': here {vendored[name]!r}, "
               f"upstream {upstream[name]!r}"
               for name in sorted(set(vendored) & set(upstream))
               if vendored[name] != upstream[name]]

    return added + removed + changed


def main() -> int:
    """Compare the two copies and report."""
    try:
        upstream = fetch_upstream()
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Could not read the upstream tokens from {UPSTREAM_URL}: {exc}")
        print("The vendored copy was NOT verified. This is a failure rather "
              "than a pass, because a check that cannot run has not run.")
        return UNREACHABLE

    drift = differences(comparable(read_vendored()), comparable(upstream))

    if not drift:
        print(f"Design tokens agree with {REPOSITORY}.")
        return AGREE

    print(f"Design tokens have drifted from {REPOSITORY}:")
    for line in drift:
        print(f"  - {line}")

    print()
    print("The design system lives in beacon-ui, so the fix is almost always to "
          "copy its tokens/colors.json over src/beacon/tokens/colors.json — not "
          "to edit the values here.")

    return DRIFTED


if __name__ == "__main__":
    sys.exit(main())
