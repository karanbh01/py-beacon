# tests/test_report.py
"""BN-97: the report block model and the PDF renderer."""
import hashlib

import pytest

from beacon.exceptions import ReportingError
from beacon.report import (
    PAGE_SIZES,
    BarChart,
    Chart,
    Header,
    PageSetup,
    ReportTemplate,
    Stat,
    StatGrid,
    Table,
    Text,
    block_from_dict,
)
from beacon.report.pdf import render
from beacon.server.store import DocumentStore

A4_WIDTH, A4_HEIGHT = PAGE_SIZES["A4"]


def factsheet() -> ReportTemplate:
    """A representative one-page report using every block type."""
    return ReportTemplate(
        template_id="factsheet-v1",
        name="Beacon Factsheet",
        page=PageSetup(size="A4"),
        blocks=[
            Header("Beacon Global Technology Index", "Monthly factsheet",
                   "31 Dec 2024"),
            StatGrid([Stat("Index level", "1,842.10", "+2.4% MTD"),
                      Stat("1Y return", "18.42%"),
                      Stat("Volatility", "15.84%"),
                      Stat("Constituents", "6")]),
            Text("Rebalanced quarterly with a 35% single-name cap.", muted=True),
            BarChart(["AAA", "BBB", "CCC"], [0.31, -0.12, 0.22],
                     title="Contribution to return"),
            Table(["Constituent", "Weight"],
                  [["Alpha Industries", "18.4%"], ["Beta Systems", "17.1%"]],
                  title="Top holdings", align_right=[1]),
            Chart(title="Index level", height=120.0),
        ])


def digest(path) -> str:
    """Fingerprint a rendered file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestPageSetup:

    def test_a4_portrait_dimensions(self):
        assert PageSetup().dimensions == (A4_WIDTH, A4_HEIGHT)

    def test_landscape_swaps_the_axes(self):
        assert PageSetup(orientation="landscape").dimensions == (A4_HEIGHT, A4_WIDTH)

    def test_content_area_excludes_both_margins(self):
        page = PageSetup(margin=50.0)

        assert page.content_width == pytest.approx(A4_WIDTH - 100.0)
        assert page.content_height == pytest.approx(A4_HEIGHT - 100.0)

    def test_an_unknown_size_is_refused(self):
        with pytest.raises(ReportingError, match="unknown page size"):
            PageSetup(size="TABLOID")

    def test_an_unknown_orientation_is_refused(self):
        with pytest.raises(ReportingError, match="unknown orientation"):
            PageSetup(orientation="sideways")

    def test_a_negative_margin_is_refused(self):
        with pytest.raises(ReportingError, match="non-negative"):
            PageSetup(margin=-5.0)

    def test_it_round_trips(self):
        page = PageSetup(size="LETTER", orientation="landscape", margin=30.0)

        assert PageSetup.from_dict(page.to_dict()) == page


class TestBlockSerialisation:
    """A template is stored as JSON and read back by a different process."""

    @pytest.mark.parametrize("block", [
        Header("Title", "Sub", "2024-12-31"),
        Text("Body text", size=10.0, muted=True),
        StatGrid([Stat("Label", "42", "+1")], columns=2),
        Table(["A", "B"], [["1", "2"]], title="T", align_right=[1]),
        BarChart(["X", "Y"], [1.0, -2.0], title="Bars", height=100.0),
        Chart(image_path=None, title="Chart", height=150.0),
    ])
    def test_every_block_round_trips(self,
                                     block):
        assert block_from_dict(block.to_dict()) == block

    def test_the_kind_discriminator_is_carried(self):
        assert Header("T").to_dict()["kind"] == "header"
        assert BarChart(["A"], [1.0]).to_dict()["kind"] == "bar_chart"

    def test_an_unknown_kind_is_refused(self):
        """Skipping it would silently drop content from a report."""
        with pytest.raises(ReportingError, match="unknown block kind"):
            block_from_dict({"kind": "hologram"})

    def test_a_block_with_no_kind_is_refused(self):
        with pytest.raises(ReportingError, match="unknown block kind"):
            block_from_dict({"title": "orphan"})

    def test_a_whole_template_round_trips(self):
        template = factsheet()

        assert ReportTemplate.from_dict(template.to_dict()) == template


class TestBlockValidation:

    def test_a_table_row_of_the_wrong_width_is_refused(self):
        with pytest.raises(ReportingError, match="row 1 has 1 cells"):
            Table(["A", "B"], [["1", "2"], ["3"]])

    def test_a_bar_chart_with_mismatched_lengths_is_refused(self):
        with pytest.raises(ReportingError, match="categories but"):
            BarChart(["A", "B"], [1.0])

    def test_an_empty_bar_chart_is_refused(self):
        with pytest.raises(ReportingError, match="at least one category"):
            BarChart([], [])

    def test_a_stat_grid_needs_at_least_one_column(self):
        with pytest.raises(ReportingError, match="at least 1"):
            StatGrid([Stat("A", "1")], columns=0)


class TestRendering:

    def test_it_writes_a_file(self,
                              tmp_path):
        destination = render(factsheet(), tmp_path / "report.pdf")

        assert destination.exists()
        assert destination.read_bytes().startswith(b"%PDF")

    def test_it_creates_missing_directories(self,
                                            tmp_path):
        destination = render(factsheet(), tmp_path / "nested" / "deep" / "r.pdf")

        assert destination.exists()

    def test_it_produces_exactly_one_page(self,
                                          tmp_path):
        content = render(factsheet(), tmp_path / "report.pdf").read_bytes()
        pages = content.count(b"/Type /Page") - content.count(b"/Type /Pages")

        assert pages == 1

    def test_two_renders_are_byte_identical(self,
                                            tmp_path):
        """The acceptance criterion.

        A PDF normally carries a creation timestamp and a random document id,
        so identical output is something the renderer has to arrange rather
        than something it gets. Without it every rerender looks like a change
        and reviewing a report diff is worthless.
        """
        first = render(factsheet(), tmp_path / "a.pdf")
        second = render(factsheet(), tmp_path / "b.pdf")

        assert digest(first) == digest(second)

    def test_a_stored_template_renders_identically(self,
                                                    tmp_path):
        """Round-tripping through storage must not change a single byte."""
        template = factsheet()
        restored = ReportTemplate.from_dict(template.to_dict())

        assert digest(render(template, tmp_path / "a.pdf")) == digest(
            render(restored, tmp_path / "b.pdf"))

    def test_different_content_gives_a_different_file(self,
                                                      tmp_path):
        """Determinism must not have flattened the content away."""
        changed = ReportTemplate(template_id="factsheet-v1", name="Beacon Factsheet",
                                 blocks=[Header("A different title")])

        assert digest(render(factsheet(), tmp_path / "a.pdf")) != digest(
            render(changed, tmp_path / "b.pdf"))

    def test_an_empty_template_still_renders(self,
                                             tmp_path):
        empty = ReportTemplate(template_id="blank", name="Blank")

        assert render(empty, tmp_path / "blank.pdf").exists()

    def test_landscape_pages_are_wider_than_tall(self,
                                                 tmp_path):
        template = ReportTemplate(
            template_id="wide", name="Wide",
            page=PageSetup(orientation="landscape"),
            blocks=[Header("Sideways")])

        content = render(template, tmp_path / "wide.pdf").read_bytes()

        assert b"842" in content  # the A4 long edge, now the width


class TestOverflow:
    """One page, and it says so rather than guessing at pagination."""

    def test_too_much_content_is_refused(self,
                                         tmp_path):
        template = ReportTemplate(
            template_id="long", name="Long",
            blocks=[Chart(title=f"Chart {index}", height=200.0)
                    for index in range(8)])

        with pytest.raises(ReportingError, match=r"only .* of the page is left"):
            render(template, tmp_path / "long.pdf")

    def test_the_message_names_the_offending_block(self,
                                                   tmp_path):
        template = ReportTemplate(
            template_id="long", name="Long",
            blocks=[Chart(height=700.0), Table(["A"], [["1"]])])

        with pytest.raises(ReportingError, match="table block needs"):
            render(template, tmp_path / "long.pdf")

    def test_a_bigger_page_fits_more(self,
                                     tmp_path):
        blocks = [Chart(title=f"Chart {index}", height=150.0) for index in range(4)]

        tight = ReportTemplate(template_id="t", name="T",
                               page=PageSetup(size="A5"), blocks=blocks)
        roomy = ReportTemplate(template_id="r", name="R",
                               page=PageSetup(size="A4"), blocks=blocks)

        with pytest.raises(ReportingError):
            render(tight, tmp_path / "tight.pdf")

        assert render(roomy, tmp_path / "roomy.pdf").exists()


class TestChartPlaceholder:

    def test_a_chart_without_an_image_still_renders(self,
                                                    tmp_path):
        """A template is designed before the charts it holds exist."""
        template = ReportTemplate(template_id="c", name="C",
                                  blocks=[Chart(title="Coming soon")])

        assert render(template, tmp_path / "c.pdf").exists()

    def test_a_missing_image_path_falls_back_to_the_placeholder(self,
                                                                tmp_path):
        """Failing here would block a layout review on an unrelated artifact."""
        template = ReportTemplate(
            template_id="c", name="C",
            blocks=[Chart(image_path=str(tmp_path / "absent.png"), title="Gone")])

        assert render(template, tmp_path / "c.pdf").exists()

    def test_a_real_image_is_drawn(self,
                                   tmp_path):
        """A one-pixel PNG, written by hand so the test needs no image library."""
        import base64

        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
            "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
        image = tmp_path / "dot.png"
        image.write_bytes(png)

        template = ReportTemplate(
            template_id="c", name="C",
            blocks=[Chart(image_path=str(image), title="Real")])
        with_image = render(template, tmp_path / "with.pdf")

        blank = ReportTemplate(template_id="c", name="C",
                               blocks=[Chart(title="Real")])
        without = render(blank, tmp_path / "without.pdf")

        assert digest(with_image) != digest(without)


class TestPersistence:
    """Templates live in the same DocumentStore as everything else."""

    def test_a_template_survives_a_store_round_trip(self,
                                                     tmp_path):
        store = DocumentStore("report_templates", root=tmp_path)
        template = factsheet()

        store.write(template.template_id, template.to_dict())
        restored = ReportTemplate.from_dict(store.read(template.template_id))

        assert restored == template

    def test_a_stored_template_renders(self,
                                       tmp_path):
        store = DocumentStore("report_templates", root=tmp_path)
        store.write("factsheet-v1", factsheet().to_dict())

        restored = ReportTemplate.from_dict(store.read("factsheet-v1"))
        output = render(restored, tmp_path / "from_store.pdf")

        assert output.read_bytes().startswith(b"%PDF")


class TestBarChartLayout:

    def test_negative_values_render(self,
                                    tmp_path):
        """Negative bars grow the other way, so a contribution chart reads right."""
        template = ReportTemplate(
            template_id="b", name="B",
            blocks=[BarChart(["Up", "Down"], [0.4, -0.6])])

        assert render(template, tmp_path / "b.pdf").exists()

    def test_all_zero_values_render(self,
                                    tmp_path):
        """The scale divisor would be zero; a minimum bar width covers it."""
        template = ReportTemplate(
            template_id="b", name="B",
            blocks=[BarChart(["A", "B"], [0.0, 0.0])])

        assert render(template, tmp_path / "b.pdf").exists()

    def test_a_chart_with_and_without_negatives_differs(self,
                                                        tmp_path):
        """A zero axis is drawn only when it is needed."""
        signed = ReportTemplate(template_id="s", name="S",
                                blocks=[BarChart(["A", "B"], [1.0, -1.0])])
        unsigned = ReportTemplate(template_id="u", name="U",
                                  blocks=[BarChart(["A", "B"], [1.0, 1.0])])

        assert digest(render(signed, tmp_path / "s.pdf")) != digest(
            render(unsigned, tmp_path / "u.pdf"))


class TestTextWrapping:

    def test_long_text_wraps_rather_than_overflowing(self,
                                                     tmp_path):
        """Measured in the real font, not guessed from a character count."""
        long_body = " ".join(["methodology"] * 200)
        template = ReportTemplate(template_id="t", name="T",
                                  blocks=[Text(long_body)])

        assert render(template, tmp_path / "t.pdf").exists()

    def test_explicit_newlines_are_honoured(self,
                                            tmp_path):
        single = ReportTemplate(template_id="a", name="A",
                                blocks=[Text("one two")])
        split = ReportTemplate(template_id="b", name="B",
                               blocks=[Text("one\ntwo")])

        assert digest(render(single, tmp_path / "a.pdf")) != digest(
            render(split, tmp_path / "b.pdf"))


class TestMissingExtra:

    def test_importing_without_reportlab_names_the_extra(self):
        """The acceptance criterion, in a subprocess with reportlab blocked."""
        import subprocess
        import sys

        script = (
            "import sys\n"
            "class Blocker:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] == 'reportlab':\n"
            "            raise ImportError(name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "import beacon.report.pdf\n"
        )

        completed = subprocess.run([sys.executable, "-c", script],
                                   capture_output=True, text=True, check=False)

        assert completed.returncode != 0
        assert 'py-beacon[pdf]' in completed.stderr
        assert "MissingDependencyError" in completed.stderr

    def test_the_block_model_needs_no_extra(self):
        """A client builds and stores templates without a PDF library."""
        import subprocess
        import sys

        script = (
            "import sys\n"
            "class Blocker:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] == 'reportlab':\n"
            "            raise ImportError(name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "from beacon.report import Header, ReportTemplate\n"
            "t = ReportTemplate('x', 'X', blocks=[Header('Hi')])\n"
            "assert ReportTemplate.from_dict(t.to_dict()) == t\n"
            "print('ok')\n"
        )

        completed = subprocess.run([sys.executable, "-c", script],
                                   capture_output=True, text=True, check=False)

        assert completed.returncode == 0, completed.stderr
        assert "ok" in completed.stdout


class TestUnrenderableBlock:

    def test_a_block_type_the_renderer_does_not_know_is_refused(self,
                                                                tmp_path):
        """A guard on the seam between the two halves of this package.

        Blocks are data and the renderer is a separate module, so a block can
        be added to the model and its drawing routine forgotten. Silently
        skipping it would drop content from a report without saying so.
        """
        from dataclasses import dataclass, field

        from beacon.report.blocks import Block

        @dataclass(frozen=True)
        class Sparkline(Block):
            kind: str = field(default="sparkline", init=False)

        template = ReportTemplate(template_id="x", name="X",
                                  blocks=[Sparkline()])

        with pytest.raises(ReportingError, match="no renderer for a sparkline"):
            render(template, tmp_path / "x.pdf")
