# src/beacon/analysis/reports/definitions.py
"""
The reports a run's analysis is read in, each at two levels: ``core``, the
statistics shown by default with others offered, and ``extended``, everything
the report covers.

    REPORTS["factsheet"].level("core").included

A level lists its statistics in sections, by category, in the order the
categories are read; each statistic is ``included`` (shown by default) or
``optional`` (offered, off by default), with the time view it is shown in by
default: ``point``, ``rolling`` or ``both``. Everything in a report's
extended view is optional for its core.
"""
# BN-398/BN-400. The reports' contents are Karan's choices of 29 October;
# their sections follow the categories until he sets them.
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalogue import CATALOGUE, CATEGORIES

CORE, EXTENDED = "core", "extended"
LEVELS = (CORE, EXTENDED)

INCLUDED, OPTIONAL = "included", "optional"
POINT, ROLLING, BOTH = "point", "rolling", "both"
VIEWS = (POINT, ROLLING, BOTH)

_FILE = Path(__file__).with_name("reports.json")


@dataclass(frozen=True)
class Entry:
    """One statistic in a report level.

    Attributes:
        id: The statistic's id.
        state: ``included`` or ``optional``.
        view: The time view it is shown in by default.
    """
    id: str
    state: str
    view: str


@dataclass(frozen=True)
class Section:
    """A level's statistics of one category, in order."""
    category: str
    label: str
    entries: tuple[Entry, ...]


@dataclass(frozen=True)
class Level:
    """A report at one level of detail."""
    name: str
    sections: tuple[Section, ...]

    @property
    def entries(self) -> list[Entry]:
        """Every statistic in the level, in order."""
        return [entry for section in self.sections for entry in section.entries]

    @property
    def included(self) -> list[str]:
        """The statistics shown by default."""
        return [entry.id for entry in self.entries if entry.state == INCLUDED]

    @property
    def optional(self) -> list[str]:
        """The statistics offered, off by default."""
        return [entry.id for entry in self.entries if entry.state == OPTIONAL]

    def entry(self,
              statistic_id: str) -> Entry | None:
        """*statistic_id*'s entry, or None when the level does not cover it."""
        return next((entry for entry in self.entries if entry.id == statistic_id), None)


@dataclass(frozen=True)
class Report:
    """One report, with its two levels."""
    id: str
    label: str
    order: int
    levels: dict[str, Level]

    def level(self,
              name: str) -> Level:
        """The report at *name* (``core`` or ``extended``).

        Raises:
            ValueError: If *name* is neither.
        """
        if name not in self.levels:
            raise ValueError(f"A report's level is {' or '.join(LEVELS)}, got {name!r}.")

        return self.levels[name]


def _level(name: str,
           chosen: dict[str, Any]) -> Level:
    """A level from its included and optional ids and their views, in
    category order and then the catalogue's."""
    states = {**dict.fromkeys(chosen[OPTIONAL], OPTIONAL), **dict.fromkeys(chosen[INCLUDED],
                                                                         INCLUDED)}
    unknown = sorted(set(states) - set(CATALOGUE.statistics))

    if unknown:
        raise ValueError(f"The {name} level names statistics the catalogue lacks: "
                         f"{', '.join(unknown)}.")

    views = chosen.get("views", {})
    sections = []

    for category, label in CATEGORIES.items():
        entries = tuple(Entry(statistic.id, states[statistic.id], views.get(statistic.id, POINT))
                        for statistic in CATALOGUE
                        if statistic.category == category and statistic.id in states)

        if entries:
            sections.append(Section(category, label, entries))

    return Level(name, tuple(sections))


def load_reports(path: Path = _FILE) -> dict[str, Report]:
    """The reports kept in *path*, in order."""
    kept = json.loads(path.read_text(encoding="utf-8"))

    return {report_id: Report(report_id, entry["label"], order,
                              {name: _level(name, entry[name]) for name in LEVELS})
            for order, (report_id, entry) in enumerate(kept.items(), start=1)}


REPORTS = load_reports()
