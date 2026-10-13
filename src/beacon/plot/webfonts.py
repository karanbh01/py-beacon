# src/beacon/plot/webfonts.py
"""
The fonts a chart's or table's HTML carries, so it is set as its image is
wherever it is opened, offline included.

Each face used is embedded once, as the file the image is drawn from: the
bundled faces (under the SIL Open Font License, which allows embedding) or a
font given as the path of its file. A family that is neither is left to the
browser.
"""
# BN-302, shared with BN-301.
import base64
import functools
from collections.abc import Iterable
from pathlib import Path

from .metrics import BUNDLED, Font, font_file


def family_name(name: str) -> str:
    """A font's family as HTML names it: a font file by its name."""
    if name.lower().endswith((".ttf", ".otf")):
        return Path(name).stem

    return name


def css_family(name: str) -> str:
    """A font as a CSS font family, with the fonts that stand in for it."""
    return f"'{family_name(name)}', 'Inter Tight', 'Inter', system-ui, sans-serif"


def font_faces(fonts: Iterable[Font]) -> str:
    """The @font-face rules for the faces text is set in, each file once,
    serving every weight that is set in it."""
    faces: dict[tuple[str, bool, Path], list[int]] = {}

    for font in fonts:
        if not embeddable(font.family):
            continue
        key = (family_name(font.family), font.italic, font_file(font))
        faces.setdefault(key, []).append(font.weight)

    return "".join(_face(family, italic, path, weights)
                   for (family, italic, path), weights in sorted(faces.items(), key=str))


def embeddable(family: str) -> bool:
    """Whether a family has a file to embed: bundled, or given as one."""
    return (family.lower().endswith((".ttf", ".otf"))
            or any(key[0] == family for key in BUNDLED))


def _face(family: str,
          italic: bool,
          path: Path,
          weights: list[int]) -> str:
    kind = "opentype" if path.suffix.lower() == ".otf" else "truetype"

    return (f"@font-face{{font-family:'{family}';font-style:{'italic' if italic else 'normal'};"
            f"font-weight:{min(weights)} {max(weights)};"
            f"src:url(data:font/{path.suffix.lower()[1:]};base64,{_encoded(path)}) "
            f"format('{kind}')}}")


@functools.cache
def _encoded(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")
