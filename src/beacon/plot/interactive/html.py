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


# What a figure that fills its frame needs: no margin round the page, and
# the page as tall as the frame.
FILL = "<style>html,body{margin:0;height:100%;overflow:hidden}</style>"


def figure_html(figure: Any,
                render: Callable[..., str],
                *args: Any,
                fonts: str | None = "embed",
                fit: bool = False,
                **kwargs: Any) -> str:
    """plotly's HTML for a figure, with the chart's fonts before it and the
    popup's script after it.

    Args:
        figure: The figure.
        render: plotly's own to_html for it.
        fonts: "embed" to carry the fonts in the page, a stylesheet's address
            to link one that has them (beacon.plot.webfonts.font_faces with
            a base), or None for neither.
        fit: Whether the figure fills the frame it is shown in rather than
            keeping its own size, as in an iframe sized by the page.
    """
    kwargs.setdefault("config", {**CONFIG, "responsive": fit})
    kwargs.setdefault("include_plotlyjs", True)
    scripts = kwargs.get("post_script") or []
    kwargs["post_script"] = [*([scripts] if isinstance(scripts, str) else scripts), SCRIPT]
    styles = ""
    if fonts == "embed":
        styles = f"<style>{font_faces(figure_fonts(figure))}</style>"
    elif fonts is not None:
        styles = f'<link rel="stylesheet" href="{fonts}">'

    if fit:
        size = figure.layout.width, figure.layout.height
        figure.update_layout(width=None, height=None, autosize=True)
        kwargs.setdefault("default_width", "100%")
        kwargs.setdefault("default_height", "100%")
        styles += FILL
        try:
            written = render(*args, **kwargs)
        finally:
            figure.update_layout(width=size[0], height=size[1], autosize=False)
    else:
        written = render(*args, **kwargs)

    if "<body>" in written:
        return written.replace("<body>", f"<body>{styles}", 1)

    return styles + written


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


def figure_fonts(figure: Any) -> set[Font]:
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
