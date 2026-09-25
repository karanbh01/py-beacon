# src/beacon/changelog.py
"""The engine's own changelog, as data.

The app (Beacon) shows the user what the engine they are running has
changed. It learns the engine's version from `/health`, so the changelog
comes from the engine for the same reason: the notes have to be for the
engine actually running, not whatever the app was built against.

An installed wheel carries a copy of `CHANGELOG.md` inside the package,
because an installed engine has no repository beside it. An editable install
reads the file at the repository root directly.

The format is Keep a Changelog: `## [version] - date`, then `### Added`,
`### Changed` and so on, each a list of `- ` items. Items stay markdown, so
the emphasis and code spans in them survive to whatever renders them.
"""
# BN-222. `CHANGELOG.md` at the repository root is the one file anybody
# edits; the wheel's copy comes from `force-include` in pyproject.toml.
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

# The copy the wheel carries, and where it sits inside the package.
PACKAGED_NAME = "CHANGELOG.md"

_RELEASE = re.compile(r"^## \[(?P<version>[^\]]+)\](?:\s*-\s*(?P<date>\S+))?")
_SECTION = re.compile(r"^### (?P<heading>.+?)\s*$")
_ITEM = re.compile(r"^- (?P<text>.*)$")
# A link reference definition, e.g. `[0.1.0]: https://...`, which closes the
# file and belongs to no section.
_LINK = re.compile(r"^\[[^\]]+\]:\s")


@dataclass(frozen=True)
class ChangelogSection:
    """One heading under a release, e.g. "Added", and its items."""
    heading: str
    items: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChangelogEntry:
    """One release: its version, its date, and what changed in it.

    `date` is None for the Unreleased entry, which has none.
    """
    version: str
    date: str | None
    sections: list[ChangelogSection] = field(default_factory=list)


def parse(text: str) -> list[ChangelogEntry]:
    """Read a Keep a Changelog document into entries, newest first.

    Prose between a release heading and its first section (a note that
    nothing has been released yet, say) is not an item and is skipped. A
    list item that wraps onto indented lines is joined back into one string.
    """
    entries: list[ChangelogEntry] = []
    section: ChangelogSection | None = None

    for line in text.splitlines():
        release = _RELEASE.match(line)

        if release is not None:
            entries.append(ChangelogEntry(version=release["version"],
                                          date=release["date"]))
            section = None
            continue

        if not entries or _LINK.match(line):
            continue

        heading = _SECTION.match(line)

        if heading is not None:
            section = ChangelogSection(heading=heading["heading"])
            entries[-1].sections.append(section)
            continue

        section = _add_line(section, line)

    return entries


def _add_line(section: ChangelogSection | None,
              line: str) -> ChangelogSection | None:
    """Fold one body line into the current section, returning it."""
    if section is None:
        return None

    item = _ITEM.match(line)

    if item is not None:
        section.items.append(item["text"].strip())
    elif line.startswith("  ") and line.strip() and section.items:
        section.items[-1] = f"{section.items[-1]} {line.strip()}"

    return section


def newer_than(entries: list[ChangelogEntry],
               since: str) -> list[ChangelogEntry]:
    """The entries released after *since*, for "what's new" in a client.

    Compared as numbers, so 0.10.0 is after 0.9.0, and a *since* this
    changelog does not list still works: an app that last saw a newer engine
    gets nothing, not the whole history. Unreleased has no number and is
    always included.

    Raises:
        ValueError: If *since* is not a dotted version such as 0.1.0.
    """
    floor = _as_numbers(since)

    if floor is None:
        raise ValueError(f"{since!r} is not a version such as 0.1.0.")

    return [entry for entry in entries
            if (number := _as_numbers(entry.version)) is None or number > floor]


def _as_numbers(version: str) -> tuple[int, ...] | None:
    """`0.1.0` as `(0, 1, 0)`, or None for anything that is not dotted digits."""
    parts = version.split(".")

    if not all(part.isdigit() for part in parts):
        return None

    return tuple(int(part) for part in parts)


def read() -> str:
    """The changelog text for this installation.

    Raises:
        FileNotFoundError: If neither the packaged copy nor the repository
            file exists, which means the wheel was built wrongly.
    """
    packaged = resources.files("beacon").joinpath(PACKAGED_NAME)

    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")

    # An editable install: the package directory is src/beacon, so the
    # repository root is two levels up.
    source = Path(__file__).resolve().parents[2] / "CHANGELOG.md"

    return source.read_text(encoding="utf-8")


def entries() -> list[ChangelogEntry]:
    """This installation's changelog, newest first."""
    return parse(read())
