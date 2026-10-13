# src/beacon/plot/tables/fonts.py
"""
The fonts a table's HTML carries, so it is set as its image is wherever it is
opened, offline included.

Each face the table uses is embedded once, as the file the image is drawn
from: the bundled faces (under the SIL Open Font License, which allows
embedding) or a font given as the path of its file. A family that is neither
is left to the browser.
"""
# BN-302.
import base64
import functools
from pathlib import Path
from typing import Any

from ..arrange import SUBTITLE_WEIGHT, TITLE_WEIGHT
from ..metrics import BUNDLED, Font, font_file
from .sizes import BOLD_WEIGHT, GROUP_WEIGHT, VALUE_WEIGHT

# The weight a notes line is set in.
NOTES_WEIGHT = 300


def family_name(name: str) -> str:
    """A font's family as the HTML names it: a font file by its name."""
    if name.lower().endswith((".ttf", ".otf")):
        return Path(name).stem

    return name


def css_family(name: str) -> str:
    """A font as a CSS font family, with the fonts that stand in for it."""
    return f"'{family_name(name)}', 'Inter Tight', 'Inter', system-ui, sans-serif"


def font_faces(table: Any) -> str:
    """The @font-face rules for every face a table's HTML sets text in."""
    faces: dict[tuple[str, bool, Path], list[int]] = {}

    for font in _used(table):
        if not _embeddable(font.family):
            continue
        key = (family_name(font.family), font.italic, font_file(font))
        faces.setdefault(key, []).append(font.weight)

    return "".join(_face(family, italic, path, weights)
                   for (family, italic, path), weights in faces.items())


def _used(table: Any) -> set[Font]:
    """The faces a table's HTML sets text in: each family at each weight and
    slant it is asked for."""
    style, model = table.style, table.model
    text, heading = style.text.font, style.heading
    used = {Font(text, VALUE_WEIGHT), Font(text, GROUP_WEIGHT),
            Font(text, style.table.header_weight), Font(text, style.table.name_weight),
            Font(heading.title_font, TITLE_WEIGHT)}

    if any(row.bold for group in model.row_groups for row in group.rows):
        used.add(Font(text, BOLD_WEIGHT))
    if style.notes.show and table._notes():
        used.add(Font(text, NOTES_WEIGHT, style.notes.italic))
    if table.subtitle:
        used.add(Font(heading.subtitle_font, SUBTITLE_WEIGHT, heading.subtitle_italic))

    return used


def _embeddable(family: str) -> bool:
    return (family.lower().endswith((".ttf", ".otf"))
            or any(key[0] == family for key in BUNDLED))


def _face(family: str,
          italic: bool,
          path: Path,
          weights: list[int]) -> str:
    """One face, serving every weight that is set in it."""
    kind = "opentype" if path.suffix.lower() == ".otf" else "truetype"

    return (f"@font-face{{font-family:'{family}';font-style:{'italic' if italic else 'normal'};"
            f"font-weight:{min(weights)} {max(weights)};"
            f"src:url(data:font/{path.suffix.lower()[1:]};base64,{_encoded(path)}) "
            f"format('{kind}')}}")


@functools.cache
def _encoded(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")
