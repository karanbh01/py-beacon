# src/beacon/plot/interactive/axes.py
"""
A panel's axes as plotly draws them, every one taken from the layout: its
domain is the panel's box, its range and ticks are the laid-out ones, and
none of plotly's own choices are left to it.
"""
# BN-301.
from dataclasses import dataclass
from typing import Any

from ..layout import AxisLayout, Box
from ..metrics import Font
from .baseline import shift
from .units import DAY_MS, PT, Paint

# Tick labels' weight, as matplotlib sets them.
TICK_WEIGHT = 300

# The space between a tick label and its axis, in points as designed.
LABEL_PAD = 3.5


@dataclass(frozen=True)
class Names:
    """A panel's plotly axis names: as a layout key ("xaxis2") and as a
    trace refers to it ("x2")."""

    x: str
    y: str
    y2: str | None = None

    @staticmethod
    def key(ref: str) -> str:
        """The layout key for an axis reference: "x2" is "xaxis2"."""
        return f"{ref[0]}axis{ref[1:]}"


def domain(box: Box,
           width: float,
           height: float) -> tuple[list[float], list[float]]:
    """A box as fractions of the figure, across and up."""
    return ([box.left / width, box.right / width], [box.bottom / height, box.top / height])


def axis(laid: AxisLayout,
         span: list[float],
         anchor: str,
         paint: Paint,
         across: bool,
         grid: bool) -> dict[str, Any]:
    """One axis: its domain, range, ticks, line and gridlines, as laid out."""
    style = paint.style
    scale = DAY_MS if laid.kind == "date" else 1.0
    low, high = laid.low * scale, laid.high * scale
    k = paint.k

    return {
        "type": "date" if laid.kind == "date" else "linear",
        "domain": span,
        "anchor": anchor,
        "range": [high, low] if laid.inverted else [low, high],
        "autorange": False,
        "fixedrange": True,
        "tickmode": "array",
        "tickvals": [tick.value * scale for tick in laid.ticks],
        "ticktext": [tick.label for tick in laid.ticks],
        # Tick labels are drawn as annotations (tick_labels), where the image
        # puts them, rather than where plotly would.
        "showticklabels": False,
        # Tick marks are drawn as shapes (tick_marks): plotly leaves out the
        # ones at an axis's two ends.
        "ticks": "",
        # The axis line too is a shape (axis_line), so it and its tick marks
        # are drawn alike and meet.
        "showline": False,
        "showgrid": grid,
        "gridcolor": paint.ink("divider"),
        "gridwidth": style.lines.grid_width * k * PT,
        "zeroline": False,
        "showspikes": False,
        "layer": "above traces",
        "hoverformat": "%-d %b %Y" if laid.kind == "date" else "",
    }


def axis_line(refs: tuple[str, str],
              across: bool,
              base: float,
              length: float,
              paint: Paint) -> list[dict[str, Any]]:
    """An axis line, centred on the plot's edge (or on zero), run on past its
    ends by half its width so two lines meet square at a corner.

    Args:
        refs: Its own reference and the other axis's, such as ("x", "y").
        across: Whether it is a horizontal axis.
        base: Where it sits on the other axis, as a share of its domain.
        length: Its length, in pixels.
        paint: The chart's inks and sizes.
    """
    axes = paint.style.axes
    if not (axes.x_line if across else axes.y_line) or length <= 0:
        return []

    width = paint.style.lines.axis_width * paint.k * PT
    over = width / 2.0 / length
    own, other = refs
    ends = ({"x0": -over, "x1": 1.0 + over, "y0": base, "y1": base} if across
            else {"x0": base, "x1": base, "y0": -over, "y1": 1.0 + over})

    return [{"type": "line", "layer": "above", "xref": f"{own if across else other} domain",
             "yref": f"{other if across else own} domain",
             "line": {"color": paint.ink("text-primary"), "width": width}, **ends}]


def tick_marks(laid: AxisLayout,
               refs: tuple[str, str],
               across: bool,
               length: float,
               base: float,
               paint: Paint) -> list[dict[str, Any]]:
    """An axis's tick marks, every one the layout has, ends included.

    Args:
        laid: The axis.
        refs: Its own reference and the other axis's, such as ("x", "y").
        across: Whether it is a horizontal axis.
        length: The panel's size across the ticks, in pixels.
        base: Where the axis line sits on the other axis, as a share of its
            domain: 0 at the plot's edge.
        paint: The chart's inks and sizes.
    """
    axes = paint.style.axes
    shown = (axes.x_ticks if across else axes.y_ticks) and laid.kind not in ("names", "fixed")
    if not shown or length <= 0:
        return []

    reach = axes.tick_length * paint.k * PT / length
    tip = base + (reach if axes.tick_direction == "in" else -reach)
    scale = DAY_MS if laid.kind == "date" else 1.0
    line = {"color": paint.ink("text-primary"),
            "width": paint.style.lines.axis_width * paint.k * PT}
    own, other = refs
    marks = []

    for tick in laid.ticks:
        at = tick.value * scale
        ends = ({"x0": at, "x1": at, "y0": base, "y1": tip} if across
                else {"x0": base, "x1": tip, "y0": at, "y1": at})
        marks.append({"type": "line", "layer": "above", "line": line,
                      "xref": own if across else f"{other} domain",
                      "yref": f"{other} domain" if across else own, **ends})

    return marks


def tick_labels(laid: AxisLayout,
                refs: tuple[str, str],
                side: str,
                base: float,
                paint: Paint) -> list[dict[str, Any]]:
    """An axis's tick labels, set where the image sets them: the style's pad
    from the axis line (and past its tick marks where they point out), each
    label hung from its baseline.

    Args:
        laid: The axis.
        refs: Its own reference and the other axis's, such as ("x", "y").
        side: Where the labels are: "bottom", "left" or "right".
        base: Where the axis line sits on the other axis, as a share of its
            domain.
        paint: The chart's inks and sizes.
    """
    if not laid.show_labels:
        return []

    axes = paint.style.axes
    ticked = (axes.x_ticks if side == "bottom" else axes.y_ticks) and laid.kind not in (
        "names", "fixed")
    pad = LABEL_PAD * paint.k * PT + (
        axes.tick_length * paint.k * PT if ticked and axes.tick_direction == "out" else 0.0)
    font = Font(paint.style.text.font, TICK_WEIGHT)
    css = {"family": paint.family, "size": paint.tick_size * PT,
           "color": paint.ink("text-primary"), "weight": TICK_WEIGHT}
    scale = DAY_MS if laid.kind == "date" else 1.0
    own, other = refs
    shown = [tick for tick in laid.ticks if tick.label]
    # One baseline for the row: each label's own would differ by a fraction
    # of a pixel, which plotly rounds into a visible step.
    words = "".join(tick.label for tick in shown)
    labels = []

    for index, tick in enumerate(shown):
        place: dict[str, Any] = {"text": tick.label, "showarrow": False, "font": css,
                                 "borderpad": 0, "textangle": -laid.rotation}
        if side == "bottom":
            place.update(x=tick.value * scale, xref=own, y=base, yref=f"{other} domain",
                         yshift=-pad)
            if laid.rotation:
                place.update(xanchor=laid.align, yanchor="top")
            else:
                place.update(xanchor="center", yanchor="bottom",
                             yshift=-pad + shift(words, paint.tick_size, font, "top"))
        else:
            va = _end(laid, index, len(shown))
            place.update(x=base, xref=f"{other} domain", y=tick.value * scale, yref=own,
                         xanchor="right" if side == "left" else "left",
                         xshift=-pad if side == "left" else pad, yanchor="bottom",
                         yshift=shift(words if va == "center_baseline" else tick.label,
                                      paint.tick_size, font, va))
        labels.append(place)

    return labels


def _end(laid: AxisLayout,
         index: int,
         count: int) -> str:
    """How a vertical label is aligned: on its tick, except a fixed scale's
    two end labels, which sit within its ends."""
    if laid.kind == "fixed" and count > 1 and index in (0, count - 1):
        return "bottom" if index == 0 else "top"

    return "center_baseline"


def origin(position: float,
           span: list[float],
           low: float,
           high: float) -> float:
    """Where an axis through zero sits on the other axis's domain."""
    share = (position - low) / ((high - low) or 1.0)

    return span[0] + min(max(share, 0.0), 1.0) * (span[1] - span[0])
