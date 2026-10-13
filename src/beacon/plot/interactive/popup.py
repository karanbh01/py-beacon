# src/beacon/plot/interactive/popup.py
"""
The popup an interactive chart shows under the pointer, as Beacon desktop's
popover draws it (beacon-ui Popover.css): a glass fill blurring what is
behind it, a half-pixel edge, a 2px radius and a soft shadow. Plotly's own
labels are SVG, which cannot blur, so the script hides them and draws this
in HTML over the chart, fed by plotly's hover events; where the script
cannot run, plotly's labels stay.

It also draws the app's crosshair: a dashed line in the muted text colour
down through the date under the pointer, with a dot on each line there, or
both ways through a scatter's point.
"""
# BN-301.
from typing import Any

from ...tokens import DARK, colour
from .units import Paint

# The glass and its edge by the page a chart sits on: Beacon's on its own
# canvases, white glass on a white page, GitHub's overlay grey on its dark
# one.
WHITE_GLASS = ("rgba(255,255,255,0.65)", "#d0d7de")
GITHUB_GLASS = ("rgba(21,27,35,0.65)", "#3d444d")
GLASS = {"white": WHITE_GLASS, "transparent-dark-axes": WHITE_GLASS,
         "github-dark": GITHUB_GLASS, "transparent-light-axes": GITHUB_GLASS}


def glass(paint: Paint,
          crosshair: str) -> dict[str, Any]:
    """What the popup is drawn in, kept in the figure's layout so its script
    finds it: the glass, its edge, the text and muted colours, the font and
    the kind of crosshair."""
    theme = paint.theme
    fill, edge = GLASS.get(theme.name, (None, None))
    if fill is None:
        mode = theme.mode
        fill, edge = _hex_rgba(colour("glass", mode)), colour("chrome-border", mode)

    return {"glass": fill, "edge": edge, "text": paint.ink("text-primary"),
            "muted": paint.ink("text-secondary"), "font": paint.family,
            "size": round(paint.tick_size * 96.0 / 72.0, 2), "crosshair": crosshair,
            "dark": theme.mode == DARK}


def _hex_rgba(value: str) -> str:
    """A #rrggbbaa token as a CSS rgba()."""
    digits = value.lstrip("#")
    red, green, blue = (int(digits[i:i + 2], 16) for i in (0, 2, 4))
    alpha = int(digits[6:8], 16) / 255.0 if len(digits) == 8 else 1.0

    return f"rgba({red},{green},{blue},{alpha:.3f})"


# Run after plotly draws the figure; "{plot_id}" is its div's id.
SCRIPT = r"""
(function () {
  var gd = document.getElementById("{plot_id}");
  if (!gd || !gd.on) return;
  var meta = ((gd.layout || {}).meta || {}).beacon;
  if (!meta) return;
  var host = gd.parentNode;
  if (getComputedStyle(host).position === "static") host.style.position = "relative";
  var css = document.createElement("style");
  // An attribute selector: plotly's ids can start with a digit, which an
  // id selector cannot.
  css.textContent = '[id="{plot_id}"] .hoverlayer{display:none!important}' +
    ".beacon-popup{position:absolute;pointer-events:none;z-index:10;padding:5px 7px;" +
    "border-radius:2px;backdrop-filter:blur(3px);-webkit-backdrop-filter:blur(3px);" +
    "box-shadow:0 4px 6px rgb(0 0 0 / 15%);white-space:nowrap;line-height:1.35}" +
    ".beacon-popup .h{font-weight:400;margin-bottom:2px}" +
    ".beacon-popup .r{display:flex;align-items:center;gap:5px;font-weight:300}" +
    ".beacon-popup .s{width:7px;height:7px;border-radius:1px;flex:none}" +
    ".beacon-popup .v{margin-left:auto;padding-left:10px;font-variant-numeric:tabular-nums}" +
    ".beacon-cross{position:absolute;pointer-events:none;z-index:9}";
  document.head.appendChild(css);

  var popup = document.createElement("div");
  popup.className = "beacon-popup";
  popup.style.cssText = "display:none;background:" + meta.glass + ";border:0.5px solid " +
    meta.edge + ";color:" + meta.text + ";font-family:" + meta.font + ";font-size:" +
    meta.size + "px";
  host.appendChild(popup);
  var marks = [];

  function clear() {
    popup.style.display = "none";
    marks.forEach(function (mark) { mark.remove(); });
    marks = [];
  }
  function place(left, top, width, height, rule) {
    var mark = document.createElement("div");
    mark.className = "beacon-cross";
    mark.style.cssText = "left:" + left + "px;top:" + top + "px;width:" + width + "px;height:" +
      height + "px;" + rule;
    host.appendChild(mark);
    marks.push(mark);
  }
  function dashed(vertical) {
    return (vertical ? "border-left" : "border-top") + ":1px dashed " + meta.muted;
  }
  function centre(point) {
    var box = point.bbox || {};
    return {x: (box.x0 + box.x1) / 2 + gd.offsetLeft, y: (box.y0 + box.y1) / 2 + gd.offsetTop};
  }
  function crosshair(points) {
    var first = points[0], at = centre(first);
    var xa = first.xaxis, ya = first.yaxis;
    if (meta.crosshair === "x" || meta.crosshair === "both") {
      place(at.x, ya._offset + gd.offsetTop, 0, ya._length, dashed(true));
    }
    if (meta.crosshair === "both") {
      place(xa._offset + gd.offsetLeft, at.y, xa._length, 0, dashed(false));
    }
    if (meta.crosshair === "none") return;
    points.forEach(function (point) {
      var kind = ((point.data.meta || {}).beacon || {}).kind;
      if (kind !== "line" && kind !== "point") return;
      var dot = centre(point), rows = JSON.parse(point.customdata[2] || "[]");
      place(dot.x - 3, dot.y - 3, 6, 6, "border-radius:50%;background:" +
        (rows.length ? rows[0][0] : meta.text));
    });
  }
  function write(points) {
    var head = "", rows = [];
    points.slice().sort(function (a, b) {
      return (((a.data.meta || {}).beacon || {}).order || 0) -
        (((b.data.meta || {}).beacon || {}).order || 0);
    }).forEach(function (point) {
      var data = point.customdata;
      if (!data) return;
      head = head || data[0];
      JSON.parse(data[2] || "[]").forEach(function (row) { rows.push(row); });
    });
    if (!rows.length) return false;
    var text = function (value) {
      var span = document.createElement("span");
      span.textContent = value;
      return span.innerHTML;
    };
    popup.innerHTML = (head ? '<div class="h">' + text(head) + "</div>" : "") +
      rows.map(function (row) {
        return '<div class="r"><span class="s" style="background:' + row[0] + '"></span>' +
          (row[1] ? "<span>" + text(row[1]) + "</span>" : "") +
          '<span class="v">' + text(row[2]) + "</span></div>";
      }).join("");
    return true;
  }
  gd.on("plotly_hover", function (event) {
    clear();
    var points = (event.points || []).filter(function (point) { return point.customdata; });
    if (!points.length || !write(points)) return;
    crosshair(points);
    popup.style.display = "block";
    var pointer = event.event, box = host.getBoundingClientRect();
    var x = (pointer ? pointer.clientX - box.left : centre(points[0]).x) + 12;
    var y = (pointer ? pointer.clientY - box.top : centre(points[0]).y) + 12;
    if (x + popup.offsetWidth > host.clientWidth) x -= popup.offsetWidth + 24;
    if (y + popup.offsetHeight > host.clientHeight) y -= popup.offsetHeight + 24;
    popup.style.left = Math.max(0, x) + "px";
    popup.style.top = Math.max(0, y) + "px";
  });
  gd.on("plotly_unhover", clear);
})();
"""
