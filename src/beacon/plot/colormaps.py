# src/beacon/plot/colormaps.py
"""
Colour maps: colours from low to high, which a value between two ends is
blended along.

`beacon_corr` is the correlation map, from the mode-independent
`raw.heatmap-*` tokens, so a correlation looks the same in every theme. The
others are built from the series palette, except viridis. A name ending in
"_r" runs the map the other way, and a list of hex colours is a map of its
own.
"""
# BN-299.
from collections.abc import Sequence

from ..tokens import raw_colours

# Colours from low to high (BN-303's maps, chosen with Karan).
MAPS: dict[str, tuple[str, ...]] = {
    "blue-red": ("#11376a", "#4a88c7", "#f2f2f2", "#f0796b", "#ba151a"),
    "blues": ("#eaf1fa", "#4a88c7", "#070f67"),
    "greens": ("#e8f5ec", "#20a464", "#245723"),
    "purples": ("#f1eaf8", "#7d4fb8", "#4b2554"),
    "greys": ("#f2f2f2", "#999999", "#262626"),
    "viridis": ("#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"),
}

CORRELATION = "beacon_corr"

# Text on a cell lighter than this is written dark; on a darker one, light.
LIGHT_ENOUGH = 0.55
DARK_INK = "#262626"
LIGHT_INK = "#ffffff"


def names() -> list[str]:
    """Every map's name."""
    return [CORRELATION, *MAPS]


def stops(choice: str | Sequence[str]) -> tuple[str, ...]:
    """A colour map's colours, from low to high.

    Args:
        choice: A map's name, "_r" on the end to reverse it, or a list of hex
            colours.

    Raises:
        ValueError: If there is no such map.
    """
    if not isinstance(choice, str):
        found = tuple(choice)
        if len(found) < 2:
            raise ValueError("a colour map needs at least two colours")
        return found

    name, reverse = (choice[:-2], True) if choice.endswith("_r") else (choice, False)

    if name == CORRELATION:
        raw = raw_colours()
        found = (raw["heatmap-low"], raw["heatmap-mid"], raw["heatmap-high"])
    elif name in MAPS:
        found = MAPS[name]
    else:
        raise ValueError(f"colormap must be one of {', '.join(names())}, a name ending "
                         f"in _r, or a list of colours, not {choice!r}")

    return tuple(reversed(found)) if reverse else found


def _rgb(colour: str) -> tuple[float, float, float]:
    value = colour.lstrip("#")
    if len(value) == 3:
        value = "".join(part * 2 for part in value)

    return (int(value[0:2], 16) / 255.0, int(value[2:4], 16) / 255.0,
            int(value[4:6], 16) / 255.0)


def colour_at(colours: Sequence[str],
              share: float) -> str:
    """The colour a share of the way along a map, from 0 at its low end to 1
    at its high end."""
    share = min(1.0, max(0.0, share))
    place = share * (len(colours) - 1)
    index = min(int(place), len(colours) - 2)
    part = place - index
    low, high = _rgb(colours[index]), _rgb(colours[index + 1])
    mixed = [a + (b - a) * part for a, b in zip(low, high, strict=True)]

    return "#" + "".join(f"{round(channel * 255):02x}" for channel in mixed)


def ink_on(colour: str) -> str:
    """Text that reads on a colour: dark on a light one, light on a dark
    one."""
    red, green, blue = _rgb(colour)

    return DARK_INK if 0.2126 * red + 0.7152 * green + 0.0722 * blue > LIGHT_ENOUGH else LIGHT_INK
