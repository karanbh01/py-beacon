# src/beacon/plot/mark.py
"""
The beta mark: the glyph Beacon desktop shows at the top left of its menu bar
and pybeacon.dev in its header, drawn beside every framed chart's heading.

Kept as the glyph's outline rather than set in a font, so drawing it needs no
font at all, and it is the same shape the application and the site draw.
"""
import functools
import re

from .._optional import require

require("matplotlib", "Charting")

from matplotlib.path import Path as Outline  # noqa: E402

# An italic beta from Playfair Display, as an outline. Copied from
# beacon-site's overrides/partials/logo-beta.html, itself from beacon-ui's
# src/renderer/src/icons/svg/logo-beta.svg, in the glyph's own coordinates
# with y up.
BETA_PATH = (
    "M14.4344 33.6673C12.4646 33.1878 10.6078 31.6051 9.60677 "
    "29.5269C8.7349 27.7204 8.20208 26.0418 5.69948 17.4252C4.40781 12.981 "
    "2.95469 7.9932 2.47031 6.31463C1.92135 4.47619 1.27552 2.63776 "
    "0.791146 1.6466L0 0H1.35625H2.72865L3.11615 0.767347C3.64896 1.80646 "
    "4.8599 5.17959 5.39271 7.11395C5.65104 7.9932 5.89323 8.77653 5.97396 "
    "8.85646C6.03854 8.92041 6.4099 8.76054 6.81354 8.48878C9.26771 6.82619 "
    "13.3526 7.36973 15.9844 9.71973C17.0661 10.6789 18.2609 12.5014 "
    "18.7292 13.9242C19.2943 15.6507 19.2781 17.9367 18.6969 "
    "19.1837C18.2448 20.1429 17.2276 21.2139 16.3557 21.6776C16.049 21.8374 "
    "15.8068 22.0133 15.8068 22.0612C15.8068 22.1252 16.2589 22.381 16.8078 "
    "22.6367C18.2286 23.2922 19.5203 24.2833 20.2307 25.2425C21.3609 "
    "26.7612 21.7484 28.8874 21.2156 30.6619C20.4891 33.0759 17.4859 "
    "34.4027 14.4344 33.6673ZM17.5828 32.5643C18.4708 31.9888 18.7937 "
    "31.2854 18.7776 29.9745C18.7453 27.4806 17.1469 24.2034 15.3708 "
    "23.0364C14.6927 22.5888 14.6604 22.5888 13.9984 22.8126C12.4807 "
    "23.3241 11.4474 23.0364 11.6734 22.1412C11.8187 21.5656 12.3677 "
    "21.4058 13.4979 21.5816C14.6766 21.7735 15.0318 21.5497 15.726 "
    "20.2228C16.1781 19.3595 16.2104 19.1517 16.2104 17.681C16.1943 15.7306 "
    "15.9359 14.8194 14.7573 12.3895C14.2083 11.2544 13.6432 10.3752 "
    "13.1911 9.92755C11.738 8.4568 9.8974 8.07313 8.23438 8.88844C7.37865 "
    "9.32007 6.39375 10.3912 6.52292 10.7588C6.5875 10.9507 6.74896 11.4622 "
    "8.94479 19.0238C11.7057 28.4718 12.4646 30.518 13.6594 31.717C14.8542 "
    "32.932 16.5172 33.2837 17.5828 32.5643Z")


@functools.cache
def beta_outline() -> Outline:
    """The beta mark as a matplotlib path, parsed once from BETA_PATH."""
    return _parse(BETA_PATH)


def _parse(d: str) -> Outline:
    """An SVG path of absolute M, L, H, C and Z commands as a matplotlib path.

    Only those commands, because the mark uses only those. Anything else
    raises rather than drawing a wrong shape.
    """
    tokens = re.findall(r"[A-Za-z]|-?\d*\.?\d+(?:e-?\d+)?", d)
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    start = current = (0.0, 0.0)
    command = ""
    position = 0

    while position < len(tokens):
        if tokens[position].isalpha():
            command = tokens[position]
            position += 1

        arity = {"M": 2, "L": 2, "H": 1, "C": 6, "Z": 0}.get(command)
        if arity is None:
            raise ValueError(f"the mark's path uses {command!r}, which this "
                             f"parser does not read")

        values = [float(token) for token in tokens[position:position + arity]]
        position += arity

        if command == "M":
            current = start = (values[0], values[1])
            vertices.append(current)
            codes.append(int(Outline.MOVETO))
            command = "L"  # further pairs after a move are lines
        elif command == "L":
            current = (values[0], values[1])
            vertices.append(current)
            codes.append(int(Outline.LINETO))
        elif command == "H":
            current = (values[0], current[1])
            vertices.append(current)
            codes.append(int(Outline.LINETO))
        elif command == "C":
            current = (values[4], values[5])
            vertices.extend([(values[0], values[1]), (values[2], values[3]), current])
            codes.extend([int(Outline.CURVE4)] * 3)
        else:
            vertices.append(start)
            codes.append(int(Outline.CLOSEPOLY))
            current = start
            command = ""

    return Outline(vertices, codes)
