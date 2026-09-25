# tests/test_data_import.py
"""BN-239: loading a user's own data from CSV files or an Excel workbook.

The layout (`beacon.data.layout`) is shared with Postgres stores, so these
checks are the checks every source of a user's own data gets.
"""
import io
import zipfile

import pandas as pd
import pytest

from beacon.data import importing, layout

MARKET = "IDENTIFIER,DATE,CLOSE\nAAA,2024-01-02,100\nAAA,2024-01-03,101\n"
REFERENCE = ("IDENTIFIER,NAME,CURRENCY,EXCHANGE,DATE_FROM\n"
             "AAA,Alpha,USD,XNYS,2020-01-01\n")


def files(folder,
          **sheets: str) -> list:
    """Write each sheet as `<name>.csv` and return the paths."""
    paths = []

    for name, text in sheets.items():
        path = folder / f"{name}.csv"
        path.write_text(text, encoding="utf-8")
        paths.append(path)

    return paths


def problems(folder,
             **sheets: str) -> list[layout.Problem]:
    """What `load_files` refuses the sheets for."""
    with pytest.raises(importing.DataImportError) as raised:
        importing.load_files(files(folder, **sheets))

    return raised.value.problems


def codes(found: list[layout.Problem]) -> list[tuple]:
    return [(problem.sheet, problem.row, problem.column, problem.code)
            for problem in found]


class TestLoading:

    def test_valid_files_load(self,
                              tmp_path):
        data = importing.load_files(files(tmp_path, market=MARKET,
                                          reference=REFERENCE))

        assert data.identifiers == ["AAA"]
        assert data.fetch_price("AAA", "2024-01-03") == 101

    def test_names_are_matched_without_regard_to_case_or_spaces(self,
                                                                tmp_path):
        """A sheet called "Corporate Actions" and a column called "close"."""
        paths = files(tmp_path,
                      Market="identifier,date,close\nAAA,2024-01-02,100\n",
                      reference=REFERENCE)
        actions = tmp_path / "Corporate Actions.csv"
        actions.write_text("Identifier,Ex Date,Type,Value,Status\n"
                           "AAA,2024-01-02,dividend,0.5,Paid\n")

        data = importing.load_files([*paths, actions])

        stored = data.corporate_actions.data
        assert list(stored["TYPE"]) == ["DIVIDEND"]
        assert list(stored["STATUS"]) == ["paid"]

    def test_fx_pairs_become_rates_the_engine_converts_with(self,
                                                            tmp_path):
        data = importing.load_files(files(
            tmp_path, market=MARKET, reference=REFERENCE,
            fx="PAIR,DATE,RATE\ngbpusd,2024-01-02,1.27\n"))

        assert data.fx_rate_on("GBP", "USD", "2024-01-02") == pytest.approx(1.27)

    def test_a_feature_without_a_type_is_imported(self,
                                                  tmp_path):
        data = importing.load_files(files(
            tmp_path, market=MARKET, reference=REFERENCE,
            features="IDENTIFIER,DATE,FIELD,VALUE\nAAA,2024-01-02,revenue,10\n"))

        assert data.feature_types() == [layout.DEFAULT_FEATURE_TYPE]

    def test_extra_reference_columns_are_kept(self,
                                              tmp_path):
        reference = ("IDENTIFIER,NAME,CURRENCY,EXCHANGE,DATE_FROM,SECTOR\n"
                     "AAA,Alpha,USD,XNYS,2020-01-01,Energy\n")
        data = importing.load_files(files(tmp_path, market=MARKET,
                                          reference=reference))

        assert "SECTOR" in data.reference_columns


class TestTheTemplate:

    def test_the_excel_template_loads_as_it_is(self,
                                               tmp_path):
        workbook = tmp_path / "template.xlsx"
        workbook.write_bytes(importing.template("xlsx"))

        data = importing.load_files([workbook])

        assert "AAA" in data.identifiers

    def test_the_csv_template_loads_as_it_is(self,
                                             tmp_path):
        zipfile.ZipFile(io.BytesIO(importing.template("csv"))).extractall(tmp_path)

        data = importing.load_files(sorted(tmp_path.glob("*.csv")))

        assert "AAA" in data.identifiers

    def test_every_sheet_has_its_columns_and_an_example_row(self):
        workbook = pd.read_excel(io.BytesIO(importing.template("xlsx")),
                                 sheet_name=None)

        for sheet in layout.SHEETS:
            assert set(sheet.required) <= set(workbook[sheet.name].columns)
            assert len(workbook[sheet.name]) == 1

    def test_an_unknown_format_is_refused(self):
        with pytest.raises(ValueError, match="template format"):
            importing.template("pdf")


class TestEveryProblemIsNamed:
    """Each names its sheet, its row as a spreadsheet shows it (the header is
    row 1), and its column."""

    def test_the_two_required_sheets(self,
                                     tmp_path):
        assert codes(problems(tmp_path, market=MARKET)) == [
            ("reference", None, None, "MISSING_SHEET")]

    def test_an_unknown_sheet(self,
                              tmp_path):
        found = problems(tmp_path, market=MARKET, reference=REFERENCE,
                         prices="A\n1\n")

        assert ("prices", None, None, "UNKNOWN_SHEET") in codes(found)

    def test_a_missing_column(self,
                              tmp_path):
        found = problems(tmp_path, market="IDENTIFIER,DATE\nAAA,2024-01-02\n",
                         reference=REFERENCE)

        assert codes(found) == [("market", None, "CLOSE", "MISSING_COLUMN")]

    def test_bad_cells_are_found_at_their_rows(self,
                                               tmp_path):
        market = ("IDENTIFIER,DATE,CLOSE,FREE_FLOAT\n"
                  "AAA,2024-01-02,100,0.9\n"       # row 2: fine
                  "AAA,not a date,101,0.9\n"       # row 3
                  "AAA,2024-01-04,abc,0.9\n"       # row 4
                  "AAA,2024-01-05,-1,1.5\n"        # row 5: two problems
                  ",2024-01-08,100,\n")            # row 6
        found = problems(tmp_path, market=market, reference=REFERENCE)

        assert codes(found) == [
            ("market", 3, "DATE", "BAD_DATE"),
            ("market", 4, "CLOSE", "BAD_NUMBER"),
            ("market", 5, "CLOSE", "OUT_OF_RANGE"),
            ("market", 5, "FREE_FLOAT", "OUT_OF_RANGE"),
            ("market", 6, "IDENTIFIER", "BLANK_VALUE")]

    def test_a_date_that_could_mean_two_days_is_refused(self,
                                                        tmp_path):
        """01/02/2024 is 1 February in one country and 2 January in another,
        so only YYYY-MM-DD is read."""
        found = problems(tmp_path,
                         market="IDENTIFIER,DATE,CLOSE\nAAA,01/02/2024,100\n",
                         reference=REFERENCE)

        assert codes(found) == [("market", 2, "DATE", "BAD_DATE")]
        assert "YYYY-MM-DD" in found[0].message

    def test_a_repeated_row(self,
                            tmp_path):
        found = problems(tmp_path,
                         market=MARKET + "AAA,2024-01-02,99\n",
                         reference=REFERENCE)

        assert codes(found) == [("market", 4, None, "DUPLICATE_ROW")]

    def test_a_name_missing_from_reference_is_reported_once(self,
                                                            tmp_path):
        """One mistake, however many rows the name has."""
        market = MARKET + "ZZZ,2024-01-02,5\nZZZ,2024-01-03,6\n"
        found = problems(tmp_path, market=market, reference=REFERENCE)

        assert codes(found) == [("market", 4, "IDENTIFIER",
                                 "UNKNOWN_IDENTIFIER")]

    def test_reference_dates_out_of_order(self,
                                          tmp_path):
        reference = ("IDENTIFIER,NAME,CURRENCY,EXCHANGE,DATE_FROM,DATE_TO\n"
                     "AAA,Alpha,USD,XNYS,2020-01-01,2019-01-01\n")
        found = problems(tmp_path, market=MARKET, reference=reference)

        assert codes(found) == [("reference", 2, "DATE_TO", "OUT_OF_RANGE")]

    def test_a_pair_that_is_not_two_currencies(self,
                                               tmp_path):
        found = problems(tmp_path, market=MARKET, reference=REFERENCE,
                         fx="PAIR,DATE,RATE\nGBP-USD,2024-01-02,1.27\n")

        assert codes(found) == [("fx", 2, "PAIR", "BAD_PAIR")]

    def test_an_unknown_action_type(self,
                                    tmp_path):
        found = problems(tmp_path, market=MARKET, reference=REFERENCE,
                         corporate_actions="IDENTIFIER,EX_DATE,TYPE,VALUE\n"
                                           "AAA,2024-01-02,BONUS,1\n")

        assert codes(found) == [("corporate_actions", 2, "TYPE", "UNKNOWN_VALUE")]

    def test_a_file_that_cannot_be_read(self,
                                        tmp_path):
        stray = tmp_path / "notes.txt"
        stray.write_text("hello")

        with pytest.raises(importing.DataImportError) as raised:
            importing.load_files([*files(tmp_path, market=MARKET,
                                         reference=REFERENCE), stray])

        assert codes(raised.value.problems) == [
            ("notes.txt", None, None, "UNREADABLE_FILE")]

    def test_a_long_list_is_capped_but_counted(self,
                                               tmp_path):
        rows = "".join(f"AAA,bad-{row},1\n" for row in range(300))
        with pytest.raises(importing.DataImportError) as raised:
            importing.load_files(files(tmp_path,
                                       market="IDENTIFIER,DATE,CLOSE\n" + rows,
                                       reference=REFERENCE))

        assert len(raised.value.problems) == layout.MAX_PROBLEMS
        assert raised.value.total == 300

    def test_findings_carry_the_shape_other_refusals_use(self,
                                                         tmp_path):
        """So a client that shows findings shows these without change."""
        with pytest.raises(importing.DataImportError) as raised:
            importing.load_files(files(tmp_path, market=MARKET))

        assert raised.value.findings == [{
            "path": "reference", "rule_id": None, "severity": "error",
            "code": "MISSING_SHEET",
            "message": "The reference sheet is required.",
            "sheet": "reference", "row": None, "column": None}]
