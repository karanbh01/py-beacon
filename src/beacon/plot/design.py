# src/beacon/plot/design.py
"""
The chart design as settings: `Style`, and the functions that choose one.

Every part of a chart's look is a named setting with Beacon's design as its
default, grouped as a chart is built: its size, its heading, its text, its
lines, its axes, its legend, its notes and its colours.

* **`Style`** is the look itself: one immutable value holding every setting.
  It changes nothing until it is used, and saves to a TOML file and loads back
  from one, so a project can keep its look in a file.
* **`configure()`** makes a look the default for every chart drawn afterwards:
  a `Style`, a preset, or settings by name, such as `configure(mark=False)`.
* **`settings()`** does the same for one `with` block, then puts the previous
  look back.
* A chart's own `style=` changes that chart alone.

The most specific wins: a chart's `style=`, then a `with settings(...)` block,
then `configure()`, then Beacon's defaults.

    from beacon.plot import Style, configure, settings

    configure(mark=False, heading_scale=1.2)
    with settings(theme="white", legend="bottom"):
        result.plot.level()

    house = Style.beacon().replace(lines={"series_width": 2.0})
    house.save("house.toml")
    configure(Style.load("house.toml"))

Standard library only, so `import beacon.plot` stays free of matplotlib.
"""
# BN-303. Designed with Karan on 2026-10-12 and 13 in the chart styles preview.
import contextlib
import dataclasses
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .groups import (
    FLAT,
    GROUPS,
    AxesSettings,
    ColorSettings,
    HeadingSettings,
    LegendSettings,
    LineSettings,
    NotesSettings,
    SizeSettings,
    TableSettings,
    TextSettings,
)
from .themes import ALIASES, THEMES

__all__ = ["FLAT", "GROUPS", "AxesSettings", "ColorSettings", "HeadingSettings",
           "LegendSettings", "LineSettings", "NotesSettings", "SizeSettings", "Style",
           "StyleArgument", "TableSettings", "TextSettings", "configure", "current", "reset",
           "resolve", "scoped", "settings", "version"]

def _theme(name: str) -> str:
    """A theme's canonical name, or ValueError."""
    resolved = ALIASES.get(name, name)

    if resolved not in THEMES:
        raise ValueError(f"theme must be one of {', '.join(THEMES)}, got {name!r}.")

    return resolved


@dataclass(frozen=True)
class Style:
    """A chart look: every setting, with Beacon's design as the default.

    Build one from a preset and change what differs:

        house = Style.beacon().replace(heading={"mark": False},
                                       lines={"series_width": 2.0})

    Attributes:
        theme: The theme: "beacon-light" (the default), "beacon-dark",
            "github-dark", "white", "transparent-dark-axes" or
            "transparent-light-axes".
        size, heading, text, lines, axes, legend, notes, colors, table: The
            settings, grouped. Each group's documentation lists its own.
    """
    theme: str = "beacon-light"
    size: SizeSettings = field(default_factory=SizeSettings)
    heading: HeadingSettings = field(default_factory=HeadingSettings)
    text: TextSettings = field(default_factory=TextSettings)
    lines: LineSettings = field(default_factory=LineSettings)
    axes: AxesSettings = field(default_factory=AxesSettings)
    legend: LegendSettings = field(default_factory=LegendSettings)
    notes: NotesSettings = field(default_factory=NotesSettings)
    colors: ColorSettings = field(default_factory=ColorSettings)
    table: TableSettings = field(default_factory=TableSettings)

    def __post_init__(self) -> None:
        object.__setattr__(self, "theme", _theme(self.theme))

    # -- presets ---------------------------------------------------------------

    @classmethod
    def beacon(cls) -> "Style":
        """Beacon's design: the default."""
        return cls()

    @classmethod
    def plain(cls) -> "Style":
        """Beacon's design for work that is not a Beacon report: no beta mark,
        on white. The accent bar and Beacon's fonts stay."""
        return cls(theme="white", heading=HeadingSettings(mark=False))

    @classmethod
    def preset(cls,
               name: str) -> "Style":
        """A preset by name: "beacon" or "plain".

        Raises:
            ValueError: If there is no such preset.
        """
        presets = {"beacon": cls.beacon, "plain": cls.plain}

        if name not in presets:
            raise ValueError(f"preset must be one of {', '.join(presets)}, got {name!r}.")

        return presets[name]()

    # -- changing --------------------------------------------------------------

    def replace(self,
                theme: str | None = None,
                **groups: Mapping[str, Any] | Any) -> "Style":
        """A copy with some settings changed, by group.

        Args:
            theme: A new theme.
            **groups: A group's name and either the settings that change in
                it, as a mapping, or a whole group object.

        Raises:
            ValueError: If a group or setting does not exist, or a value is
                not allowed.
        """
        changes: dict[str, Any] = {} if theme is None else {"theme": theme}

        for name, value in groups.items():
            if name not in GROUPS:
                raise ValueError(f"There is no {name!r} group; the groups are {', '.join(GROUPS)}.")

            current = getattr(self, name)

            if isinstance(value, GROUPS[name]):
                changes[name] = value
                continue

            if not isinstance(value, Mapping):
                raise ValueError(f"{name}= takes a mapping of its settings, got {value!r}.")

            known = {f.name for f in dataclasses.fields(GROUPS[name])}
            unknown = sorted(set(value) - known)
            if unknown:
                raise ValueError(f"{name} has no setting {', '.join(map(repr, unknown))}; "
                                 f"its settings are {', '.join(sorted(known))}.")

            changes[name] = dataclasses.replace(current, **value)

        return dataclasses.replace(self, **changes)

    def updated(self,
                **settings: Any) -> "Style":
        """A copy with settings changed by their short names, as configure()
        takes them: `style.updated(mark=False, scale=1.2)`.

        Raises:
            ValueError: If a name is not a setting, or a value is not allowed.
        """
        groups: dict[str, dict[str, Any]] = {}
        theme = settings.pop("theme", None)

        for name, value in settings.items():
            if name not in FLAT:
                raise ValueError(f"{name!r} is not a chart setting. The settings are: "
                                 f"theme, {', '.join(sorted(FLAT))}.")
            group, key = FLAT[name]
            groups.setdefault(group, {})[key] = value

        return self.replace(theme=theme, **groups)

    # -- files -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Every setting, by group. Unset (None) settings are left out."""
        out: dict[str, Any] = {"theme": self.theme}

        for name in GROUPS:
            values = {k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in dataclasses.asdict(getattr(self, name)).items() if v is not None}
            out[name] = values

        return out

    @classmethod
    def from_dict(cls,
                  data: Mapping[str, Any]) -> "Style":
        """A style from to_dict()'s form; settings left out keep their
        defaults.

        Raises:
            ValueError: If a group or setting does not exist, or a value is
                not allowed.
        """
        data = dict(data)
        theme = data.pop("theme", None)
        unknown = sorted(set(data) - set(GROUPS))

        if unknown:
            raise ValueError(f"There is no {', '.join(map(repr, unknown))} group; the groups are "
                             f"theme, {', '.join(GROUPS)}.")

        return cls().replace(theme=theme, **data)

    def save(self,
             path: str | Path) -> Path:
        """Write this style to a TOML file, to load again with Style.load().

        Returns:
            Path: Where it was written.
        """
        target = Path(path)
        lines = [f"theme = {_toml(self.theme)}"]

        for name, values in self.to_dict().items():
            if name == "theme" or not values:
                continue
            lines += ["", f"[{name}]",
                      *(f"{key} = {_toml(value)}" for key, value in values.items())]

        target.write_text("\n".join(lines) + "\n", encoding="utf-8")

        return target

    @classmethod
    def load(cls,
             path: str | Path) -> "Style":
        """A style from a TOML file that Style.save() wrote, or one written by
        hand in the same form.

        Raises:
            ValueError: If the file is not valid TOML or holds settings that
                do not exist.
        """
        try:
            data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as error:
            raise ValueError(f"{path} is not valid TOML: {error}") from error

        return cls.from_dict(data)


def _toml(value: Any) -> str:
    """One value in TOML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


# -- the current style -----------------------------------------------------------

# The style in force, and a number bumped on every change, so the renderer
# re-applies a style only when it changed (beacon.plot.style.apply).
_STATE: dict[str, Any] = {"current": Style(), "version": 0}


def current() -> Style:
    """The style charts are drawn in now."""
    chosen: Style = _STATE["current"]
    return chosen


def version() -> int:
    """A number that changes whenever the current style does."""
    number: int = _STATE["version"]
    return number


def _set(style: Style) -> None:
    _STATE["current"] = style
    _STATE["version"] += 1


def configure(style: Style | str | Path | None = None,
              *,
              preset: str | None = None,
              **settings: Any) -> Style:
    """Make a look the default for every chart drawn afterwards.

    Settings given by name change the current style rather than replacing it,
    so `configure(mark=False)` and then `configure(scale=1.2)` keep both.

    Args:
        style: A Style, or the path of a TOML file Style.save() wrote, to
            start from instead of the current style.
        preset: "beacon" or "plain", to start from instead.
        **settings: Settings by their short names, such as mark=False,
            heading_scale=1.2, legend="bottom" or theme="white".

    Returns:
        Style: The style now in force.

    Raises:
        TypeError: If both a style and a preset are given.
        ValueError: If a setting does not exist or a value is not allowed.
    """
    if style is not None and preset is not None:
        raise TypeError("configure() takes a style or a preset, not both.")

    if isinstance(style, str | Path):
        style = Style.load(style)

    if style is not None:
        base = style
    elif preset is not None:
        base = Style.preset(preset)
    else:
        base = current()
    _set(base.updated(**settings) if settings else base)

    return current()


def reset() -> Style:
    """Go back to Beacon's design.

    Returns:
        Style: The default style, now in force.
    """
    _set(Style())

    return current()


@contextlib.contextmanager
def settings(style: Style | str | Path | None = None,
             *,
             preset: str | None = None,
             **changes: Any) -> Iterator[Style]:
    """A look for one `with` block; the previous one comes back after it.

        with settings(mark=False, theme="white"):
            result.plot.level()

    Takes what configure() takes.

    Yields:
        Style: The style in force inside the block.
    """
    before = current()

    try:
        yield configure(style, preset=preset, **changes)
    finally:
        _set(before)


# What a chart's style= takes: a Style, settings by short name, or None.
StyleArgument = Style | Mapping[str, Any] | None


@contextlib.contextmanager
def scoped(style: Style) -> Iterator[Style]:
    """*style* in force while one chart is drawn.

    Unlike settings(), it leaves the version alone, so drawing the chart does
    not put the style into matplotlib's rcParams for good: the chart applies
    it for itself (beacon.plot.paint).
    """
    before = current()
    _STATE["current"] = style

    try:
        yield style
    finally:
        _STATE["current"] = before


def resolve(style: Style | Mapping[str, Any] | None) -> Style:
    """The style a single chart is drawn in, from its style= argument: a
    Style, settings by short name over the current style, or None for the
    current style.

    Raises:
        ValueError: If a setting does not exist or a value is not allowed.
    """
    if style is None:
        return current()

    if isinstance(style, Style):
        return style

    return current().updated(**dict(style))
