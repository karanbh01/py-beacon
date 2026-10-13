# src/beacon/plot/interactive/html.py
"""
An interactive chart's HTML: plotly's, with the fonts the chart is set in
embedded and the popup's script run once the figure is drawn. plotly.js is
included by default, so the file works offline.
"""
# BN-301.
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..metrics import Font
from ..webfonts import font_faces
from .popup import SCRIPT

# The weights the chart's own text is set in: its ticks and labels, and its
# headings in a grid and its legend.
CHART_WEIGHTS = (300, 400)

# Without the toolbar, zoom or scroll: the chart is the layout, as the
# image is, and the popup is what it adds.
CONFIG = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False,
          "responsive": False}


def figure_html(figure: Any,
                render: Callable[..., str],
                *args: Any,
                **kwargs: Any) -> str:
    """plotly's HTML for a figure, with the chart's fonts before it and the
    popup's script after it."""
    kwargs.setdefault("config", CONFIG)
    kwargs.setdefault("include_plotlyjs", True)
    scripts = kwargs.get("post_script") or []
    kwargs["post_script"] = [*([scripts] if isinstance(scripts, str) else scripts), SCRIPT]
    written = render(*args, **kwargs)
    fonts = f"<style>{font_faces(_fonts(figure))}</style>"

    if "<body>" in written:
        return written.replace("<body>", f"<body>{fonts}", 1)

    return fonts + written


def write_figure_html(html: str,
                      file: Any,
                      beside: bool = False) -> None:
    """Write HTML to a path or a file opened for writing, and plotly.js
    beside a path when *beside*."""
    if not isinstance(file, str | Path):
        file.write(html)
        return

    path = Path(file)
    path.write_text(html, encoding="utf-8")
    script = path.parent / "plotly.min.js"
    if beside and not script.exists():
        from plotly.offline import get_plotlyjs

        script.write_text(get_plotlyjs(), encoding="utf-8")


def _fonts(figure: Any) -> set[Font]:
    """The faces the figure's text is set in: its ticks and labels in the
    chart's font, and each piece of its frame in its own."""
    used = {Font(_name(figure.layout.font.family), weight) for weight in CHART_WEIGHTS}

    for annotation in figure.layout.annotations:
        font = annotation.font
        if font is not None and font.family:
            weight = font.weight if isinstance(font.weight, int) else 400
            used.add(Font(_name(font.family), weight, font.style == "italic"))

    return used


def _name(family: str) -> str:
    """A CSS font family's first name."""
    return family.split(",", maxsplit=1)[0].strip().strip("'\"")
