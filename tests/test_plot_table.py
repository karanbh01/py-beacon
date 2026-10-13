# tests/test_plot_table.py
"""BN-302: Table, its content, its geometry and its HTML."""
import re

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from beacon.plot import Table, reset
from beacon.plot.tables.draw import nearest
from beacon.plot.tables.geometry import measure
from beacon.plot.tables.model import (
    MISSING,
    Fill,
    column_groups,
    formatter,
    heat_fill,
    over,
)
from beacon.plot.ticks import MINUS

DATA = pd.DataFrame(
    {"1Y": [8.42, 7.91, -2.35, 11.2],
     "3Y": [5.12, 4.80, 0.40, None],
     "Sharpe": [0.61, 0.58, -0.12, 0.52],
     "AUM": [1250, 640, 95, 2310]},
    index=["Fund", "Index", "Peer low", "Growth"])

GROUPS = {"Return": ["1Y", "3Y"], "Risk": ["Sharpe"]}

ROW_GROUPS = {"Ours": ["Fund", "Index"], "Peers": ["Peer low", "Growth"]}


@pytest.fixture(autouse=True)
def fresh():
    reset()
    yield
    reset()


def cells(table: Table, row: str) -> list:
    found = [one for group in table.model.row_groups for one in group.rows if one.name == row]

    return list(found[0].cells)


def texts(table: Table) -> list[str]:
    return [text.text for text in measure(table).texts]


class TestFormats:

    def test_negative_written_with_a_true_minus(self):
        assert formatter(None, "minus", False)(-1.234) == f"{MINUS}1.23"

    def test_negative_in_brackets(self):
        assert formatter("{:.1f}", "brackets", False)(-1.25) == "(1.2)"

    def test_value_rounding_to_zero_has_no_sign(self):
        assert formatter("{:.1f}", "minus", False)(-0.04) == "0.0"

    def test_missing_value_is_a_dash(self):
        write = formatter(None, "minus", False)

        assert write(None) == MISSING
        assert write(float("nan")) == MISSING

    def test_whole_numbers_keep_no_decimals_and_separate_thousands(self):
        assert formatter(None, "minus", True)(12500) == "12,500"

    def test_mapping_format(self):
        write = formatter({"decimals": 1, "percent": True, "prefix": "~"}, "minus", False)

        assert write(0.1234) == "~12.3%"

    def test_function_format(self):
        assert formatter(lambda value: f"{value:.0f}bp", "minus", False)(12.4) == "12bp"

    def test_text_passes_through(self):
        assert formatter(None, "minus", False)("n/a") == "n/a"

    def test_group_format_applies_to_its_columns(self):
        table = Table(DATA, groups=GROUPS, formats={"Return": "{:.1f}"})

        assert [cell.text for cell in cells(table, "Fund")[:2]] == ["8.4", "5.1"]


class TestGroups:

    def test_groups_by_count(self):
        headings, found = column_groups(["a", "b", "c"], {"First": 2, "Second": 1})

        assert headings == ("First", "Second")
        assert found == {"a": 0, "b": 0, "c": 1}

    def test_group_naming_a_missing_column_raises(self):
        with pytest.raises(ValueError, match="not a column"):
            column_groups(["a"], {"First": ["b"]})

    def test_columns_know_where_their_group_starts_and_ends(self):
        columns = Table(DATA, groups=GROUPS).model.columns

        assert [(column.first, column.last) for column in columns[:3]] == [
            (True, False), (False, True), (True, True)]
        assert columns[3].group is None

    def test_row_groups_in_order_with_the_rest_after(self):
        table = Table(DATA, row_groups={"Peers": ["Peer low"]})

        assert [group.label for group in table.model.row_groups] == ["Peers", ""]
        assert [row.name for row in table.model.row_groups[1].rows] == ["Fund", "Index", "Growth"]

    def test_row_group_naming_a_missing_row_raises(self):
        with pytest.raises(ValueError, match="not a row"):
            Table(DATA, row_groups={"Peers": ["Nobody"]})


class TestCells:

    def test_signed_columns_coloured_by_sign(self):
        table = Table(DATA, groups=GROUPS, signed=["Return"])
        colours = [cell.colour for cell in cells(table, "Peer low")]

        assert colours == ["danger", "success", "text-primary", "text-primary"]

    def test_highlighted_row_is_bold(self):
        table = Table(DATA, highlight=["Index"])

        assert all(cell.bold for cell in cells(table, "Index"))
        assert not any(cell.bold for cell in cells(table, "Fund"))

    def test_series_becomes_one_column(self):
        table = Table(DATA["1Y"])

        assert [column.name for column in table.model.columns] == ["1Y"]

    def test_empty_table_raises(self):
        with pytest.raises(ValueError, match="nothing to draw"):
            Table(pd.DataFrame())

    def test_not_a_frame_raises(self):
        with pytest.raises(TypeError):
            Table([1, 2, 3])

    def test_unknown_negatives_raises(self):
        with pytest.raises(ValueError, match="negatives"):
            Table(DATA, negatives="red")


class TestHeatmap:

    def test_one_column(self):
        table = Table(DATA, heatmap={"column": "Sharpe"})

        assert table.model.heat.columns == (2,)
        assert cells(table, "Fund")[2].fill is not None
        assert cells(table, "Fund")[0].fill is None

    def test_a_group_shares_one_scale(self):
        table = Table(DATA, groups=GROUPS, heatmap={"group": "Return", "scale": "group"})
        fills = [cells(table, "Growth")[0].fill, cells(table, "Peer low")[0].fill]

        # The group's largest and smallest values sit at the map's two ends.
        assert fills[0] is not None and fills[1] is not None
        assert fills[0].colour != fills[1].colour

    def test_missing_value_has_no_fill(self):
        table = Table(DATA, groups=GROUPS, heatmap={"group": "Return"})

        assert cells(table, "Growth")[1].fill is None

    def test_values_hidden(self):
        table = Table(DATA, heatmap={"column": "Sharpe", "values": False})

        assert cells(table, "Fund")[2].hidden
        assert "0.61" not in texts(table)

    def test_naming_a_missing_column_or_group_raises(self):
        with pytest.raises(ValueError, match="column"):
            Table(DATA, heatmap={"column": "Nope"})
        with pytest.raises(ValueError, match="group"):
            Table(DATA, heatmap={"group": "Nope"})

    def test_centre_makes_the_scale_even(self):
        table = Table(DATA, heatmap={"column": "1Y", "center": 0, "fade": True,
                                     "colormap": "blue-red"})
        fill = cells(table, "Fund")[0].fill

        # 8.42 of a scale running -11.2 to 11.2.
        assert fill is not None
        assert fill.alpha == pytest.approx(8.42 / 11.2, abs=1e-6)

    def test_fade_takes_the_background_at_the_neutral_point(self):
        fill = heat_fill(0.0, -1.0, 1.0, ("#0000ff", "#ffffff", "#ff0000"), True, True)

        assert fill.alpha == 0.0
        assert over(fill, "#123456") == "#123456"

    def test_one_way_map_fades_to_its_deep_end(self):
        fill = heat_fill(1.0, 0.0, 1.0, ("#ffffff", "#000080"), True, False)

        assert fill == Fill("#000080", 1.0)

    def test_missing_cell_borrows_its_nearest_neighbour(self):
        red, blue = Fill("#ff0000"), Fill("#0000ff")

        assert nearest([red, None, None, blue]) == [red, red, blue, blue]


class TestPresets:

    def rules(self, table: Table) -> int:
        return len(measure(table).rules)

    def test_plain_draws_no_rules(self):
        assert self.rules(Table(DATA, groups=GROUPS, preset="plain")) == 0

    def test_lines(self):
        # Two group rules, the header rule, three between rows, the bottom rule.
        assert self.rules(Table(DATA, groups=GROUPS)) == 7

    def test_boxed(self):
        # Two group rules and the box's four sides.
        assert self.rules(Table(DATA, groups=GROUPS, preset="boxed")) == 6

    def test_a_rule_can_be_changed_after_its_preset(self):
        table = Table(DATA, preset="plain", style={"table_bottom_rule": True})

        assert self.rules(table) == 1

    def test_boxed_groups_are_kept_apart(self):
        table = Table(DATA, groups=GROUPS, preset="boxed",
                      style={"table_box": False, "table_group_boxes": True})
        verticals = sorted({rule.x0 for rule in measure(table).rules if rule.x0 == rule.x1})

        # Return's right side and Risk's left side, a narrow gap apart.
        assert len(verticals) == 4
        assert verticals[2] - verticals[1] == pytest.approx(2 * 3.0 / 72.0, rel=0.05)


class TestRowGroupHeadings:

    def test_heading_centred_beside_its_rows(self):
        geometry = measure(Table(DATA, row_groups=ROW_GROUPS, row_group_headings=True))
        by_text = {text.text: text for text in geometry.texts}
        ours, fund, index = by_text["Ours"], by_text["Fund"], by_text["Index"]

        assert ours.ha == "center"
        assert ours.y == pytest.approx((fund.y + index.y) / 2.0)
        assert ours.x < fund.x

    def test_headings_hidden_by_default(self):
        assert "Ours" not in texts(Table(DATA, row_groups=ROW_GROUPS))

    def test_rules_across_the_table_start_at_the_row_names(self):
        geometry = measure(Table(DATA, row_groups=ROW_GROUPS, row_group_headings=True))
        ours = next(text.x for text in geometry.texts if text.text == "Ours")
        across = [rule for rule in geometry.rules if rule.y0 == rule.y1]

        # Header, three between rows (one between the groups) and bottom; only
        # the border between the headings starts in their column.
        assert len([rule for rule in across if rule.x0 > ours]) == 5
        assert len([rule for rule in across if rule.x0 < ours]) == 1

    def test_border_between_headings_stops_short_of_the_rows_rule(self):
        geometry = measure(Table(DATA, row_groups=ROW_GROUPS, row_group_headings=True))
        ours = next(text.x for text in geometry.texts if text.text == "Ours")
        border = [rule for rule in geometry.rules if rule.y0 == rule.y1 and rule.x0 < ours]
        beside = next(rule for rule in geometry.rules
                      if rule.y0 == border[0].y0 and rule is not border[0])

        assert len(border) == 1
        assert not any(rule.x0 == rule.x1 for rule in geometry.rules)
        # A gap of one and a half cell paddings to the rule between the rows.
        assert beside.x0 - border[0].x1 == pytest.approx(4.5 / 72.0, abs=1e-6)

    def test_row_names_take_the_column_headings_colour(self):
        grouped = measure(Table(DATA, row_groups=ROW_GROUPS, row_group_headings=True))
        plain = measure(Table(DATA, row_groups=ROW_GROUPS))
        colour = {text.text: text.colour for text in grouped.texts}

        assert colour["Fund"] == colour["1Y"] == "text-secondary"
        assert colour["Ours"] == "text-primary"
        assert next(text.colour for text in plain.texts if text.text == "Fund") == "text-primary"

    def test_row_names_coloured_in_html(self):
        html = Table(DATA, row_groups=ROW_GROUPS, row_group_headings=True).to_html()

        assert re.search(r"td\.n\{color:#", html)

    def test_plain_headings_have_no_rule(self):
        geometry = measure(Table(DATA, row_groups=ROW_GROUPS, row_group_headings=True,
                                 preset="plain"))

        assert geometry.rules == []


class TestNameSpace:

    def names_left(self, **style) -> float:
        geometry = measure(Table(DATA, style=style or None))
        rule = next(rule for rule in geometry.rules if rule.y0 == rule.y1)
        name = next(text for text in geometry.texts if text.text == "Peer low")

        return name.x - rule.x0

    def test_names_column_is_as_wide_as_its_names_need(self):
        table = Table(DATA)
        geometry = measure(table)
        texts_by = {text.text: text for text in geometry.texts}
        rule = next(rule for rule in geometry.rules if rule.y0 == rule.y1)

        # The widest name sits the setting's 3pt in from the table's left.
        widest = texts_by["Peer low"]
        from beacon.plot import metrics
        from beacon.plot.metrics import Font
        width = metrics.width("Peer low", widest.size, Font(widest.font.family,
                                                            widest.font.weight))
        assert widest.x - width - rule.x0 == pytest.approx(3.0 / 72.0, abs=1e-6)

    def test_space_is_a_setting(self):
        assert self.names_left(table_name_space=12) == pytest.approx(
            self.names_left() + 9.0 / 72.0, abs=1e-6)

    def test_space_in_html(self):
        assert "padding-left:12.00pt" in Table(DATA, style={"table_name_space": 12}).to_html()


class TestFrame:

    def test_notes_carry_the_source(self):
        table = Table(DATA, title="T", source="Synthetic data", notes="Net of fees.")

        assert measure(table).layout.notes.text.startswith("Notes: Source: Synthetic data.")

    def test_wide_table_widens_the_figure(self):
        wide = pd.DataFrame({f"Column {i}": [1.0] for i in range(14)}, index=["Row"])

        assert measure(Table(wide)).layout.width > 6.0

    def test_drawn_and_saved(self, tmp_path):
        target = Table(DATA, title="Saved").save(tmp_path / "table.png")

        assert target.exists() and target.stat().st_size > 0

    def test_faint_heat_cells_keep_the_text_colour(self):
        table = Table(DATA, heatmap={"column": "1Y", "center": 0, "fade": True,
                                     "colormap": "blue-red"}, style={"theme": "dark"})
        figure = table.figure
        colours = {text.get_text(): text.get_color() for text in figure.texts}

        # Peer low (-2.35) is faint on that scale; Growth (11.2) is not.
        assert colours[f"{MINUS}2.35"] == colours["Fund"]
        assert colours["11.20"] != colours["Fund"]


class TestHtml:

    def html(self, **options) -> str:
        return Table(DATA, title="Heading", groups=GROUPS, **options).to_html()

    def test_frame_and_headings(self):
        html = self.html(subtitle="Below")

        assert "Heading" in html and "Below" in html and "<svg" in html
        assert len(re.findall(r'<th class="c[^"]*" data-col=', html)) == 4

    def test_values_carry_their_numbers_for_sorting(self):
        html = self.html()

        assert 'data-v="8.42"' in html
        assert 'data-v=""' in html

    def test_each_row_group_is_its_own_body_so_sorting_keeps_it_together(self):
        html = self.html(row_groups=ROW_GROUPS)

        assert html.count("<tbody>") == 2
        assert "rows(body).sort" in html

    def test_row_group_heading_spans_its_rows(self):
        html = self.html(row_groups=ROW_GROUPS, row_group_headings=True)

        assert '<td class="rg" rowspan="2">Ours</td>' in html
        assert "insertBefore(label" in html

    def test_plain_has_no_rules(self):
        css = self.html(preset="plain").split("</style>")[0]

        assert not re.search(r"--[tblr]:inset", css)
        assert "linear-gradient" not in css

    def test_boxed_draws_no_rule_under_the_last_row_name(self):
        css = self.html(preset="boxed").split("</style>")[0]

        assert "tr.r:last-child>td:not(.gap)" not in css
        assert "tr.r:last-child>td.v{--b:inset" in css

    def test_lines_rules_between_rows_and_under_the_table(self):
        css = self.html().split("</style>")[0]

        assert "tr.r+tr.r>td:not(.gap){--t:inset" in css
        assert re.search(r"tr\.r:last-child>td:not\(\.gap\):not\(\.rg\)\{--b:inset", css)

    def test_rules_take_no_room(self):
        css = self.html().split("</style>")[0]

        assert "border:0;box-shadow:var(--t" in css
        assert not re.search(r"border-(top|bottom|left|right):", css)

    def test_columns_take_the_image_widths(self):
        table = Table(DATA, title="Heading", groups=GROUPS)
        widths = re.findall(r'<col style="width:([\d.]+)pt">', table.to_html())

        assert [float(width) for width in widths] == pytest.approx(
            [width * 72.0 for width in measure(table).widths], abs=0.01)

    def test_bundled_fonts_embedded_once_each(self):
        html = self.html(subtitle="Below", notes="Net.")
        faces = re.findall(r"@font-face\{font-family:'([^']+)';font-style:(\w+);"
                           r"font-weight:([\d ]+);src:url\(data:font/ttf;base64,", html)

        assert sorted(faces) == [("Inter Tight", "italic", "300 300"),
                                 ("Inter Tight", "normal", "300 300"),
                                 ("Inter Tight", "normal", "400 400"),
                                 ("Source Serif 4", "italic", "350 350")]

    def test_unbundled_font_left_to_the_browser(self):
        html = Table(DATA, style={"font": "Helvetica"}).to_html()

        assert "@font-face{font-family:'Helvetica'" not in html

    def test_colour_set_on_the_table_itself(self):
        html = self.html()
        table_id = re.search(r'<table id="([^"]+)"', html).group(1)

        assert re.search(rf"#{table_id}{{[^}}]*color:#", html)

    def test_heatmap_fill_and_ink(self):
        html = Table(DATA, heatmap={"column": "Sharpe"}).to_html()

        assert re.search(r'data-fill="rgba\(\d+, \d+, \d+, 1\.000\)"', html)

    def test_blend_both_marks_stand_ins(self):
        html = Table(DATA, groups=GROUPS, heatmap={"group": "Return", "shading": "blended",
                                                   "blend": "both"}).to_html()

        assert 'data-blend="both"' in html
        assert html.count("data-mix=") == 8

    def test_notebook_shows_html(self):
        table = Table(DATA)

        assert table._repr_html_().startswith("<style>")


class TestReportBlock:

    def test_chart_block_of_a_table(self, tmp_path):
        from beacon.report.blocks import Chart

        block = Chart.of(Table(DATA, title="In a report"), str(tmp_path / "t.png"),
                         title="Caption", height=150.0)

        assert block.image_path.endswith("t.png") and block.height == 150.0
        assert block.to_dict()["kind"] == "chart"
