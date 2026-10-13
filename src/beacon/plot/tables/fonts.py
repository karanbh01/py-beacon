# src/beacon/plot/tables/fonts.py
"""
The faces a table's HTML sets text in, for beacon.plot.webfonts to embed.
"""
# BN-302.
from typing import Any

from ..arrange import SUBTITLE_WEIGHT, TITLE_WEIGHT
from ..metrics import Font
from ..webfonts import css_family, family_name
from ..webfonts import font_faces as faces_of
from .sizes import BOLD_WEIGHT, GROUP_WEIGHT, VALUE_WEIGHT

# The weight a notes line is set in.
NOTES_WEIGHT = 300

__all__ = ["css_family", "family_name", "font_faces"]


def font_faces(table: Any) -> str:
    """The @font-face rules for every face a table's HTML sets text in."""
    return faces_of(_used(table))


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
