# src/beacon/plot/tables/css.py
"""
A table's look in HTML, as a style block scoped to its id: rules, boxes and
stripes are drawn by where a row is rather than written on each cell, so
they stay in place when the rows are sorted.
"""
# BN-302.
from typing import Any

from .fonts import css_family
from .sizes import BOLD_WEIGHT, GROUP_WEIGHT, LINE, ROW_RULE_GAP, VALUE_WEIGHT

# Rules are inset shadows rather than borders, so they take no room, as the
# image draws them: a cell sets the edges it has rules on, and its shadow
# draws them all.
EDGES = ("t", "b", "l", "r")

# The first and last rows of each row group, and of the whole table.
FIRST = "tbody > tr.r:first-child"
LAST = "tbody > tr.r:last-child"
FIRST_ALL = "tbody:first-of-type > tr.r:first-child"
LAST_ALL = "tbody:last-of-type > tr.r:last-child"


def table_css(writer: Any) -> str:
    """The style block for one table, from the writer that writes it."""
    settings, sizes, rules = writer.settings, writer.sizes, writer.rules
    me = f"#{writer.id}"
    header = writer.ink(settings.header_color or "text-secondary")
    rule = writer.ink(settings.rule_color) if settings.rule_color else header
    row = writer.ink(settings.row_rule_color or "divider")
    stripe = writer.ink(settings.stripe_color or "divider")
    pv, ph = sizes.row_pad, sizes.cell_pad
    heavy, light, box = (sizes.rule, rule), (sizes.row_rule, row), (sizes.box, rule)
    shadows = ",".join(f"var(--{side},0 0 #0000)" for side in EDGES)
    header_align = settings.number_align if settings.header_align == "numbers" else "center"

    css = [
        # Set on the table itself, with its colour below: in quirks mode a
        # table takes neither its parent's font nor its line height.
        f"{me}{{border-collapse:collapse;border-spacing:0;font-family:{css_family(writer.style.text.font)};"
        f"font-weight:{VALUE_WEIGHT};line-height:{LINE};font-synthesis:none;"
        f"width:100%;font-size:{sizes.body:.2f}pt;font-variant-numeric:tabular-nums;"
        f"color:{writer.ink('text-primary')}}}",
        f"{me} th,{me} td{{padding:{pv:.2f}pt {ph:.2f}pt;white-space:nowrap;"
        f"font-weight:inherit;border:0;box-shadow:{shadows}}}",
        # The row names' column takes only the width it needs.
        f"{me} .n{{text-align:{settings.name_align};font-weight:{settings.name_weight};"
        f"width:0;padding-left:{sizes.name_space:.2f}pt}}",
        f"{me} .rg,{me} .rgh{{width:0}}",
        # Under row group headings the row names take the column headings'
        # colour, so the headings stand out from them.
        *([f"{me} td.n{{color:{header}}}"] if writer.side else []),
        f"{me} td.v{{text-align:{settings.number_align};font-weight:{VALUE_WEIGHT}}}",
        f"{me} tr.b>td{{font-weight:{BOLD_WEIGHT}}}",
        f"{me} th.c{{text-align:{header_align};font-weight:{settings.header_weight};"
        f"font-size:{sizes.head:.2f}pt;color:{header};cursor:pointer}}",
        f"{me} th.grp{{padding:{pv:.2f}pt {ph:.2f}pt {0.5 * pv:.2f}pt;text-align:center;"
        f"font-weight:{GROUP_WEIGHT};font-size:{sizes.group:.2f}pt}}",
        f"{me} td.rg{{text-align:center;vertical-align:middle;font-weight:{GROUP_WEIGHT};"
        f"font-size:{sizes.group:.2f}pt}}",
        f"{me} .gap{{width:{2 * ph:.2f}pt;padding:0}}",
        # The space between a row group's heading and the rows' rules.
        f"{me} .rgap{{width:{ROW_RULE_GAP * ph:.2f}pt;min-width:{ROW_RULE_GAP * ph:.2f}pt}}",
        f"{me} td.rg{{padding:0 {ROW_RULE_GAP * ph:.2f}pt}}",
        f"{me} tbody.space td{{height:{2 * pv:.2f}pt;padding:0}}",
        f"{me} tr.s>td:not(.rg){{background:{stripe}}}",
    ]
    css += _rules(me, rules, writer.side, writer.grouped, heavy, light, ph, sizes.row_rule, row)
    css += _boxes(me, rules, settings.group_box_scope, box)

    return "".join(css)


def edge(side: str,
         line: tuple[float, str]) -> str:
    """One edge's rule, as the custom property a cell's shadow reads."""
    width, colour = line
    # At least a device pixel, so a hairline does not fade on a plain screen.
    drawn = f"max({width:.2f}pt,1px)"
    offset = {"t": f"0 {drawn}", "b": f"0 calc(-1 * {drawn})", "l": f"{drawn} 0",
              "r": f"calc(-1 * {drawn}) 0"}[side]

    return f"--{side}:inset {offset} 0 0 {colour}"


def _rules(me: str,
           rules: dict[str, bool],
           side: bool,
           grouped: bool,
           heavy: tuple[float, str],
           light: tuple[float, str],
           ph: float,
           weight: float,
           colour: str) -> list[str]:
    """The rules under the headings, between rows, under the table and
    between column groups."""
    css = []

    if rules["group_rules"]:
        line = f"linear-gradient({colour},{colour}) no-repeat"
        css.append(f"{me} th.grp:not(:empty){{background:{line} right bottom/"
                   f"calc(100% - {3 * ph:.2f}pt) {weight:.2f}pt}}")
        # Between row groups' headings, stopping short of the rows' rules:
        # the spacer cell after the heading has no rule.
        css.append(f"{me} tbody+tbody td.rg{{{edge('t', light)}}}")
    if rules["header_rule"]:
        css.append(f"{me} tr.header>th:not(.gap):not(.rgh){{{edge('b', heavy)}}}")
    if rules["row_rules"]:
        css.append(f"{me} tr.r+tr.r>td:not(.gap){{{edge('t', light)}}}")
        if side or not grouped:
            css.append(f"{me} tbody:not(.space)+tbody>tr.r:first-child>td:not(.gap):not(.rg)"
                       f"{{{edge('t', light)}}}")
    if rules["bottom_rule"]:
        css.append(f"{me} {LAST_ALL}>td:not(.gap):not(.rg){{{edge('b', heavy)}}}")
    if rules["column_rules"]:
        css.append(f"{me} .gs{{{edge('l', light)}}}")

    return css


def _boxes(me: str,
           rules: dict[str, bool],
           scope: str,
           box: tuple[float, str]) -> list[str]:
    """The boxes round the values, round each column group, or round each
    row group. They come after the rules, so where both fall on one edge the
    box is drawn."""
    css = []

    if rules["box"]:
        css.append(f"{me} td.v.first{{{edge('l', box)}}}{me} td.v.last{{{edge('r', box)}}}")
        css.append(f"{me} {FIRST_ALL}>td.v{{{edge('t', box)}}}")
        css.append(f"{me} {LAST_ALL}>td.v{{{edge('b', box)}}}")

    if not (rules["group_boxes"] or rules["row_boxes"]):
        return css

    left, right = (".gf", ".gl") if rules["group_boxes"] else (".first", ".last")
    first, last = (FIRST, LAST) if rules["row_boxes"] else (FIRST_ALL, LAST_ALL)
    css.append(f"{me} td.v{left}{{{edge('l', box)}}}{me} td.v{right}{{{edge('r', box)}}}")
    css.append(f"{me} {last}>td.v{{{edge('b', box)}}}")

    if rules["group_boxes"] and not rules["row_boxes"] and scope == "headings":
        css.append(f"{me} th.c{left}{{{edge('l', box)}}}{me} th.c{right}{{{edge('r', box)}}}")
        css.append(f"{me} th.c{{{edge('t', box)}}}")
    else:
        css.append(f"{me} {first}>td.v{{{edge('t', box)}}}")

    return css
