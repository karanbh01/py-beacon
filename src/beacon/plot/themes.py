# src/beacon/plot/themes.py
"""
The chart themes: the background a chart is drawn on, and the inks that read
on it.

    beacon-light            Beacon's light canvas (also "light")
    beacon-dark             Beacon's dark canvas (also "dark")
    github-dark             GitHub's dark page, with its greys for the text
    white                   white, with Beacon's light-mode inks
    transparent-dark-axes   no background, dark inks: for a light page
    transparent-light-axes  no background, light inks: for a dark page

Every theme takes its colours from the design tokens of one Beacon mode and
changes only what its background needs: GitHub's dark theme its text greys,
and the transparent ones the background itself. So a chart in any theme is
still drawn in Beacon's accent, signs and series colours.

A chart records its theme when it is made, so its own inks (a marker, a sign,
the frame) follow the theme even when the background, being transparent, says
nothing about it.
"""
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..tokens import DARK, LIGHT, colour


@dataclass(frozen=True)
class Theme:
    """A chart theme.

    Attributes:
        name: What `beacon.plot.use()` takes.
        style: The matplotlib style it is registered as.
        mode: LIGHT or DARK: which set of Beacon tokens its inks come from.
        canvas: The background colour; None draws no background at all.
        overrides: Token colours this theme replaces.
    """

    name: str
    style: str
    mode: str
    canvas: str | None
    overrides: Mapping[str, str] = field(default_factory=dict)

    @property
    def transparent(self) -> bool:
        """Whether the theme draws no background."""
        return self.canvas is None

    def colour(self,
               token: str) -> str:
        """One colour in this theme: "none" for a transparent canvas.

        "period", marked periods' boxes, is the theme's own where it has one
        and its divider otherwise.
        """
        if token == "canvas":
            return self.canvas if self.canvas is not None else "none"

        if token == "period":
            return self.overrides.get("period") or colour("divider", self.mode)

        return self.overrides.get(token) or colour(token, self.mode)

    def series(self) -> tuple[str, ...]:
        """The series colours, in the order series take them."""
        return SERIES[self.mode]


# GitHub's dark page (its canvas.default) and the text greys it sets on it
# (fg.default and fg.muted), with its border greys for the lines.
# The series palette, in order: blue, deep blue, magenta, pink, lime, green,
# olive, red and oxblood, then black (white on a dark canvas) and five greys
# for context lines such as peers. Orange is no longer a default. Chart colours
# rather than design tokens: the tokens are beacon-ui's (beacon.tokens), and
# these are offered to it to adopt.
# BN-303, chosen by Karan on 2026-10-13. The dark red is still to confirm.
SERIES: dict[str, tuple[str, ...]] = {
    LIGHT: ("#4a88c7", "#070f67", "#cc0088", "#ff70cf", "#adbc10", "#20a464", "#404f25",
            "#f0200d", "#682107",
            "#000000", "#262626", "#4d4d4d", "#737373", "#999999", "#bfbfbf"),
    DARK: ("#6fa7dc", "#0072a3", "#beacec", "#7d59d9", "#6bcf9a", "#ccf403", "#09714e",
           "#e0474b", "#ed8a83",
           "#ffffff", "#f2f2f2", "#cccccc", "#a6a6a6", "#808080", "#595959"),
}

# Grey 4: unsigned bars and columns.
GREY_4 = 13

GITHUB_DARK = {
    "text-primary": "#e6edf3",
    "text-secondary": "#9198a1",
    "text-muted": "#9198a1",
    "border": "#3d444d",
    "divider": "#262c36",
}

THEMES: dict[str, Theme] = {
    theme.name: theme for theme in (
        # Marked periods' boxes have their own colour where the divider is too
        # faint on that canvas (BN-303).
        Theme("beacon-light", "beacon", LIGHT, colour("canvas", LIGHT), {"period": "#e7dabb"}),
        Theme("beacon-dark", "beacon-dark", DARK, colour("canvas", DARK)),
        Theme("github-dark", "beacon-github-dark", DARK, "#0d1117", GITHUB_DARK),
        Theme("white", "beacon-white", LIGHT, "#ffffff", {"period": "#e6e6e6"}),
        Theme("transparent-dark-axes", "beacon-transparent-dark-axes", LIGHT, None,
              {"period": "#e6e6e6"}),
        Theme("transparent-light-axes", "beacon-transparent-light-axes", DARK, None,
              {"period": "#4d4d4d"}),
    )
}

# The modes' own names, which `use()` has always taken.
ALIASES = {LIGHT: "beacon-light", DARK: "beacon-dark"}

DEFAULT = "beacon-light"

# The theme `use()` last applied, which new charts are drawn in.
_STATE = {"current": DEFAULT}

# The theme each chart was made in, kept by figure. Weak, so a closed figure
# is not kept alive by having once been drawn.
_BY_FIGURE: "weakref.WeakKeyDictionary[Any, Theme]" = weakref.WeakKeyDictionary()


def theme_named(name: str) -> Theme:
    """The theme called *name*, or a mode's own name.

    Raises:
        ValueError: If there is no such theme.
    """
    found = THEMES.get(ALIASES.get(name, name))

    if found is None:
        raise ValueError(f"theme must be one of {', '.join(THEMES)}, not {name!r}")

    return found


def set_current(name: str) -> Theme:
    """Make *name* the theme new charts are drawn in."""
    theme = theme_named(name)
    _STATE["current"] = theme.name

    return theme


def current() -> Theme:
    """The theme `beacon.plot.use()` last applied."""
    return THEMES[_STATE["current"]]


def remember(figure: Any,
             facecolor: Any) -> Theme:
    """Record the theme a new figure is drawn in, read off the background
    the style gave it, and return it."""
    theme = for_background(facecolor)
    _BY_FIGURE[figure] = theme

    return theme


def of_figure(figure: Any) -> Theme | None:
    """The theme a figure was made in, when it was made here."""
    return _BY_FIGURE.get(figure)


def for_background(facecolor: Any) -> Theme:
    """The theme a background belongs to.

    The current theme when it is the one that background came from, then any
    theme with exactly that canvas; otherwise light or dark by brightness, so
    a figure drawn on matplotlib's own white still gets Beacon's light inks.
    """
    from matplotlib.colors import to_rgba

    red, green, blue, alpha = to_rgba(facecolor)
    present = current()

    if alpha == 0.0:
        return present if present.transparent else _transparent_by_ink()

    if present.canvas is not None and to_rgba(present.canvas) == (red, green, blue, alpha):
        return present

    for theme in THEMES.values():
        if theme.canvas is not None and to_rgba(theme.canvas) == (red, green, blue, alpha):
            return theme

    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue

    return THEMES["beacon-dark" if luminance < 0.5 else "beacon-light"]


def _transparent_by_ink() -> Theme:
    """A transparent theme chosen by the style's text colour: light text means
    the light-axes theme, for a dark page."""
    import matplotlib
    from matplotlib.colors import to_rgb

    red, green, blue = to_rgb(matplotlib.rcParams["text.color"])
    light_text = 0.2126 * red + 0.7152 * green + 0.0722 * blue >= 0.5

    return THEMES["transparent-light-axes" if light_text else "transparent-dark-axes"]
