# src/beacon/plot/interactive/baseline.py
"""
Where text sits up and down, as the image sets it.

The layout anchors text as matplotlib does, by the box its ink fills (at least
as tall as "lp"). Plotly anchors the box the font's own ascent and descent
fill, which is taller. So each piece of text is placed by its baseline: the
layout's anchor gives the baseline, and plotly's box is hung below it by the
font's descent.
"""
# BN-301.
import functools

from .. import metrics
from ..metrics import Font
from .units import PX


@functools.cache
def descent(font: Font) -> float:
    """How far a font's box reaches below its baseline, as a share of its
    size."""
    from fontTools.ttLib import TTFont

    face = TTFont(str(metrics.font_file(font)), lazy=True)

    return -float(face["hhea"].descent) / float(face["head"].unitsPerEm)


def shift(words: str,
          size: float,
          font: Font,
          va: str) -> float:
    """How far up from the layout's anchor to put the bottom of plotly's box,
    in pixels, so the baseline falls where the image's does."""
    box = metrics.box(words or "lp", size, font)
    baseline = {"baseline": 0.0, "top": -box.ascent, "bottom": box.descent,
                "center": (box.descent - box.ascent) / 2.0,
                "center_baseline": -box.ascent / 2.0}[va]

    return (baseline - descent(font) * size / 72.0) * PX
