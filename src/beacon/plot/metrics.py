# src/beacon/plot/metrics.py
"""
How wide and how tall a chart's text is set, read from the font files.

A chart's layout (beacon.plot.layout) is worked out before anything is drawn,
and the same layout serves every renderer, so text cannot be measured by
drawing it. It is measured here from the fonts themselves: each glyph's
advance and the exact bounds of its outline, the way matplotlib sets a text
path. Glyphs are set at their advances with no kerning, as matplotlib sets
them for fonts whose kerning lives in GPOS rather than a kern table, which
covers the bundled fonts.

The bundled fonts are found by family; any other font is given as the path
of its file (.ttf or .otf). A family that is neither is measured as Inter
Tight, with a warning, since there is no file to read.
"""
import functools
import logging
from dataclasses import dataclass
from pathlib import Path

from .._optional import require

logger = logging.getLogger(__name__)

FONTS = Path(__file__).parent / "fonts"

# The bundled faces by family, weight and slant.
BUNDLED = {
    ("Inter Tight", 400, False): "InterTight-Regular.ttf",
    ("Inter Tight", 300, False): "InterTight-Light.ttf",
    ("Inter Tight", 300, True): "InterTight-LightItalic.ttf",
    ("Source Serif 4", 350, True): "SourceSerif4-Italic350.ttf",
}

FALLBACK = "Inter Tight"

# Weights by name, as matplotlib reads them.
WEIGHTS = {"ultralight": 100, "light": 300, "normal": 400, "regular": 400,
           "book": 400, "medium": 500, "roman": 500, "semibold": 600,
           "demibold": 600, "demi": 600, "bold": 700, "heavy": 800,
           "extra bold": 800, "black": 900}

# The text matplotlib measures every line's height against, so a line of
# digits is as tall as one with an ascender and a descender.
REFERENCE_TEXT = "lp"


@dataclass(frozen=True)
class Font:
    """A face to set text in: a family (or a font file's path), a weight and
    a slant."""

    family: str
    weight: int = 400
    italic: bool = False


@dataclass(frozen=True)
class TextBox:
    """The box a line of text is set in, in inches: its ink width, its height
    above the baseline and its depth below it."""

    width: float
    ascent: float
    descent: float

    @property
    def height(self) -> float:
        """From the bottom of the descent to the top of the ascent."""
        return self.ascent + self.descent


def weight_of(weight: str | int | float) -> int:
    """A weight as a number, from a name such as "light" or a number."""
    if isinstance(weight, str):
        return WEIGHTS.get(weight.lower(), 400)

    return int(weight)


def font_file(font: Font) -> Path:
    """The file a face is read from.

    A path is itself; a bundled family is its nearest face, matching the slant
    first and then the weight, as matplotlib chooses. Anything else is
    measured as Inter Tight.
    """
    if font.family.lower().endswith((".ttf", ".otf")):
        return Path(font.family)

    family = font.family if any(key[0] == font.family for key in BUNDLED) else None

    if family is None:
        _warn_unmeasurable(font.family)
        family = FALLBACK

    faces = [key for key in BUNDLED if key[0] == family]
    best = min(faces, key=lambda key: (key[2] != font.italic, abs(key[1] - font.weight)))

    return FONTS / BUNDLED[best]


@functools.cache
def _warn_unmeasurable(family: str) -> None:
    """Say once that a family has no file to measure."""
    logger.warning("The font %r is not bundled and was not given as a file, so "
                   "its text is measured as %s; give the path of its .ttf or "
                   ".otf file to measure it.", family, FALLBACK)


class _Face:
    """One font file's glyph advances and outline bounds, in font units."""

    def __init__(self, path: Path):
        require("fontTools", "Chart layout")

        from fontTools.pens.boundsPen import BoundsPen
        from fontTools.ttLib import TTFont

        font = TTFont(str(path), lazy=True)
        self.units = float(font["head"].unitsPerEm)
        self.cmap = font.getBestCmap()
        self.glyphs = font.getGlyphSet()
        self.advances = {name: float(advance)
                         for name, (advance, _) in font["hmtx"].metrics.items()}
        self.missing = self.cmap.get(ord("n"), ".notdef")
        self._pen = BoundsPen
        self._bounds: dict[str, tuple[float, float, float, float] | None] = {}

    def bounds(self,
               name: str) -> tuple[float, float, float, float] | None:
        """A glyph's outline bounds, None for one with no outline (a space).
        Worked out on first use, from the outline's curves themselves."""
        if name not in self._bounds:
            pen = self._pen(self.glyphs)
            self.glyphs[name].draw(pen)
            self._bounds[name] = pen.bounds

        return self._bounds[name]

    def glyph(self, character: str) -> str:
        """The glyph that sets *character*, or a stand-in when the font has
        none (matplotlib draws those from another font)."""
        return str(self.cmap.get(ord(character), self.missing))


@functools.cache
def _face(path: str) -> _Face:
    """A font file, read once."""
    return _Face(Path(path))


def _ink(text: str,
         face: _Face) -> tuple[float, float, float, float] | None:
    """The bounds of a line's ink in font units: left, bottom, right, top,
    with the glyphs set at their advances from x = 0."""
    x = 0.0
    found: list[tuple[float, float, float, float]] = []

    for character in text:
        name = face.glyph(character)
        bounds = face.bounds(name)

        if bounds is not None:
            left, bottom, right, top = bounds
            found.append((x + left, bottom, x + right, top))

        x += face.advances.get(name, 0.0)

    if not found:
        return None

    return (min(box[0] for box in found), min(box[1] for box in found),
            max(box[2] for box in found), max(box[3] for box in found))


def width(text: str,
          size: float,
          font: Font) -> float:
    """How wide *text* is set at *size* points, in inches: the width of its
    ink, the widest line's when it has several."""
    face = _face(str(font_file(font)))
    widest = 0.0

    for line in text.split("\n"):
        ink = _ink(line, face)
        if ink is not None:
            widest = max(widest, ink[2] - ink[0])

    return widest / face.units * size / 72.0


def box(text: str,
        size: float,
        font: Font) -> TextBox:
    """The box one line of *text* is set in, at *size* points.

    Its height is at least the reference text's, so a row of labels lines up
    whatever letters each one has, as matplotlib sets them.
    """
    face = _face(str(font_file(font)))
    scale = size / face.units / 72.0
    ink = _ink(text, face)
    reference = _ink(REFERENCE_TEXT, face)
    assert reference is not None  # every face sets an l and a p

    top = max(reference[3], ink[3] if ink else 0.0)
    bottom = min(reference[1], ink[1] if ink else 0.0)
    wide = (ink[2] - ink[0]) if ink else 0.0

    return TextBox(width=wide * scale, ascent=top * scale, descent=-bottom * scale)
