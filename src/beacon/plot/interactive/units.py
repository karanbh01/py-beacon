# src/beacon/plot/interactive/units.py
"""
The figure's units and inks: inches and points as plotly's pixels, dates as
its milliseconds, and the colours, fonts and sizes a chart is drawn in.
"""
# BN-301.
from dataclasses import dataclass, field

from .. import style as sizes
from ..colours import resolve
from ..design import Style
from ..panels import SCALE
from ..themes import Theme, theme_named
from ..webfonts import css_family

# Pixels to the inch, as a browser counts them, and to the point.
PX = 96.0
PT = PX / 72.0

# A plotly date axis counts milliseconds; the layout counts days.
DAY_MS = 86_400_000.0


@dataclass
class Paint:
    """A chart's inks, font and sizes, in its theme and style."""

    style: Style
    theme: Theme = field(init=False)
    family: str = field(init=False)
    k: float = field(init=False)
    tick_size: float = field(init=False)

    def __post_init__(self) -> None:
        self.theme = theme_named(self.style.theme)
        self.family = css_family(self.style.text.font)
        self.k = SCALE * self.style.size.scale
        self.tick_size = sizes.tick_label_size()

    def ink(self, name: str) -> str:
        """A colour by name, as a CSS colour."""
        found = resolve(name, self.theme, self.style)

        return "rgba(0,0,0,0)" if found == "none" else found

    @property
    def canvas(self) -> str | None:
        """The background, or None where the theme draws none."""
        found = resolve("canvas", self.theme, self.style)

        return None if found == "none" else found


def rgba(colour: str,
         alpha: float) -> str:
    """A hex colour at an opacity."""
    value = colour.lstrip("#")
    if len(value) == 3:
        value = "".join(part * 2 for part in value)
    red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))

    return f"rgba({red},{green},{blue},{alpha:.3f})"
