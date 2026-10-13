# src/beacon/plot/colours.py
"""
Colours by name, resolved in a theme and a style.

A chart's layout and marks carry colours as names, so one chart draws in any
theme: a design token such as "accent" or "text-primary", "palette-N" for the
Nth series colour, "accent-bar" for the bar beside the title, "period-label"
for a marked period's label, or a literal "#rrggbb" a caller chose. Each
renderer resolves the names here, in the theme it draws and the style's own
colours, which replace the theme's where they are set.
"""
# BN-299.
from .design import Style
from .themes import Theme

# The tokens a style's colours replace, and the setting that replaces each.
STYLE_COLOURS = {"success": "up", "danger": "down", "text-primary": "text",
                 "text-secondary": "muted", "period": "period", "canvas": "background"}

PALETTE = "palette-"


def palette(index: int) -> str:
    """The name of the series colour at *index*, counting from 0."""
    return f"{PALETTE}{index + 1}"


def series_colours(theme: Theme,
                   style: Style) -> tuple[str, ...]:
    """The series colours in order: the style's, or the theme's palette."""
    return style.colors.series or theme.series()


def resolve(name: str,
            theme: Theme,
            style: Style) -> str:
    """A colour by name, as a colour a renderer can draw.

    Args:
        name: A token, "palette-N", "accent-bar", "period-label", a literal
            "#rrggbb", or "none".
        theme: The theme drawn in.
        style: The style drawn in.

    Returns:
        str: A hex colour, or "none".
    """
    colours = style.colors

    if name.startswith("#") or name == "none":
        return name
    if name.startswith(PALETTE):
        series = series_colours(theme, style)
        return series[(int(name.removeprefix(PALETTE)) - 1) % len(series)]
    if name == "accent-bar":
        return colours.accent_bar or theme.colour("accent")
    if name == "period-label":
        return colours.period_label or resolve("text-secondary", theme, style)

    setting = STYLE_COLOURS.get(name)
    chosen = getattr(colours, setting) if setting else None

    return chosen or theme.colour(name)
