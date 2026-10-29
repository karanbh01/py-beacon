# src/beacon/server/analysis_schemas.py
"""
The wire shapes of a run's analysis: the request a backtest takes, the
analytics block a run gives, the catalogue of what can be asked for, and a
large table by column.
"""
# BN-398/BN-400.
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

COMPUTED, NOT_COMPUTED = "computed", "not_computed"


class AnalysisRequestSpec(BaseModel):
    """What a run's analysis should give: the reports wanted, each at a
    level, with statistics added or taken out, their time views, and every
    choice the analysis makes. `GET /analysis/options` lists the reports,
    statistics and settings."""
    model_config = ConfigDict(extra="forbid")
    reports: dict[str, str] = Field(
        default_factory=lambda: {"factsheet": "core"},
        description="Each report wanted, by id, with its level: 'core' or 'extended'.")
    add: dict[str, list[str]] = Field(
        default_factory=dict,
        description="By report, optional statistics to show, by id.")
    remove: dict[str, list[str]] = Field(
        default_factory=dict,
        description="By report, included statistics not to show, by id.")
    time_views: dict[str, dict[str, str]] = Field(
        default_factory=dict,
        description="By report, statistic ids with the time view to show them in: "
                    "'point', 'rolling' or 'both'.")
    statistics: list[str] = Field(
        default_factory=list,
        description="Statistics wanted outside any report, by id.")
    options: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Every choice the analysis makes, by section and setting, as "
                    "`GET /analysis/options` describes them; a setting left out takes its "
                    "default.")


class StatisticRef(BaseModel):
    """One statistic a report shows, and how."""
    id: str = Field(description="The statistic's id, a key of the block's `statistics` "
                                "or `unavailable`.")
    view: str = Field(description="The time view it is shown in: 'point', 'rolling' or "
                                  "'both'.")


class ReportSelection(BaseModel):
    """What a report shows, in order."""
    level: str = Field(description="The level asked for: 'core' or 'extended'.")
    statistics: list[StatisticRef] = Field(description="Its statistics, in reading order.")


class RollingValue(BaseModel):
    """A statistic over each trailing window."""
    window: int = Field(description="The days each window covers.")
    step: str = Field(description="How often a window ends: 'daily' or 'month_end'.")
    table: dict[str, Any] = Field(
        description="Dates down, the point figure's names across: `{index, columns, data}`, "
                    "a list of values a row.")


class StatisticValue(BaseModel):
    """One statistic's value."""
    kind: str = Field(description="'figures' (`point` maps names to numbers, names of "
                                  "several parts joined with '/'), 'series' (`point` is "
                                  "`{index, name, data}`) or 'table' (`point` is `{index, "
                                  "columns, data}`, a list of values a row).")
    point: Any = Field(description="Its value over the analysis window, in its kind's shape.")
    summary: dict[str, Any] | None = Field(
        default=None, description="For a table, the figures that sum it up (its totals, "
                                  "its residual), by name.")
    rolling: RollingValue | None = Field(
        default=None, description="Its value over each trailing window, when its time view "
                                  "asks for it.")


class UnavailableReason(BaseModel):
    """Why a statistic was not worked out for the run."""
    code: str = Field(description="A short code, such as 'no_index' or 'no_futures'; "
                                  "`GET /analysis/options` labels each.")
    reason: str = Field(description="A sentence saying why, fit to show as it is.")


class TableRef(BaseModel):
    """A large table of the run, read on its own."""
    rows: int = Field(description="How many rows it has.")
    path: str = Field(description="Where to read it: `GET` this path, with `Accept: "
                                  "application/vnd.apache.arrow.stream` for Arrow.")


class AnalyticsBlock(BaseModel):
    """A run's analysis: each statistic once, and each report as a list of
    statistic ids."""
    request: dict[str, Any] = Field(description="The request, every default filled in.")
    reports: dict[str, ReportSelection] = Field(description="Each report asked for, by id.")
    statistics: dict[str, StatisticValue] = Field(
        description="Each statistic worked out, by id. Labels, units and decimals are in "
                    "`GET /analysis/options`.")
    unavailable: dict[str, UnavailableReason] = Field(
        description="Each statistic asked for that the run cannot give, by id.")
    active_against: str | None = Field(
        description="The comparator relative and active figures are measured against.")
    groupings: list[str] = Field(
        default_factory=list,
        description="The reference fields the run kept for its names, which "
                    "`options.attribution.grouping` can name.")
    timings: dict[str, float] = Field(
        description="Seconds each part took: 'tables', then 'performance', 'portfolio', "
                    "'risk' and 'attribution' as asked for.")
    tables: dict[str, TableRef] = Field(
        default_factory=dict, description="The run's large tables, by name, read on their own.")


class AnalyticsView(BaseModel):
    """Response of `GET /strategies/{strategy_id}/analytics`."""
    status: str = Field(description="'computed', or 'not_computed' for a run made without "
                                    "an analysis, or before 0.10.0, which the analytics "
                                    "rerun gives.")
    analytics: AnalyticsBlock | None = Field(description="The run's analysis; null when "
                                                         "not computed.")
    reason: str | None = Field(default=None, description="Why it was not computed.")


class CodedColumn(BaseModel):
    """A column of text sent once per distinct value."""
    values: list[Any] = Field(description="Each distinct value.")
    codes: list[int] = Field(description="A row's value, as its position in `values`.")


class ColumnarTable(BaseModel):
    """A large table by column: each column a list of values, or, for
    repeated text, its distinct values and a code a row."""
    length: int = Field(description="How many rows it has.")
    columns: dict[str, CodedColumn | list[Any]] = Field(
        description="Each column by name, in order: a list of values a row, or a coded "
                    "column.")
    constant: dict[str, Any] = Field(
        default_factory=dict, description="Columns the same on every row, each sent once.")
    decimals: dict[str, int] = Field(
        default_factory=dict, description="The decimals each number column is rounded to.")


class StatisticSpec(BaseModel):
    """One statistic, as the catalogue lists it."""
    id: str = Field(description="What requests and stored choices name it by.")
    label: str = Field(description="Its name, as shown.")
    category: str = Field(description="What it is a figure of; reports are sectioned by it.")
    kind: str = Field(description="'figures', 'series' or 'table'.")
    part: str = Field(description="The analysis that works it out: 'performance', "
                                  "'portfolio', 'risk' or 'attribution'.")
    description: str = Field(description="One line saying what it measures.")
    unit: str | None = Field(description="What its figures are measured in; null for none.")
    decimals: int | None = Field(description="How many decimals it is shown to.")
    rolling: str | None = Field(
        description="What rolling it costs ('light', 'moderate' or 'heavy'); null when it "
                    "does not roll.")
    step: str | None = Field(description="How often its windows end unless asked otherwise.")
    settings: list[str] = Field(description="The settings it reads, as 'section.setting'.")
    figure_names: list[str] = Field(
        description="What its figure names are made of, part by part, each a key of the "
                    "catalogue's `names`.")
    columns: list[dict[str, Any]] = Field(
        description="For a table, its columns in order: name, label and unit.")
    signed: list[str] = Field(
        description="Which of its figures are returns or differences, whose sign says better "
                    "or worse, by the last part of a figure's, a column's or a summary "
                    "figure's name; '*' for every number; empty for none.")


class ReportEntrySpec(BaseModel):
    """One statistic in a report level."""
    id: str = Field(description="The statistic's id.")
    state: str = Field(description="'included' (shown by default) or 'optional' (offered).")
    view: str = Field(description="Its default time view: 'point', 'rolling' or 'both'.")


class ReportSectionSpec(BaseModel):
    """A level's statistics of one category, in order."""
    label: str = Field(description="The section's heading.")
    order: int = Field(description="Its place in the level, ascending.")
    statistics: list[ReportEntrySpec] = Field(description="Its statistics, in order.")


class ReportLevelSpec(BaseModel):
    """A report at one level."""
    sections: list[ReportSectionSpec] = Field(description="Its sections, in reading order.")


class ReportSpec(BaseModel):
    """One report, at both levels."""
    id: str = Field(description="What requests name it by.")
    label: str = Field(description="Its name, as shown.")
    order: int = Field(description="Its place among the reports, ascending.")
    levels: dict[str, ReportLevelSpec] = Field(
        description="'core' and 'extended'; each lists its sections in full.")


class TableSpec(BaseModel):
    """A large table of a run."""
    name: str = Field(description="What the tables path names it by.")
    label: str = Field(description="Its name, as shown.")
    columns: list[dict[str, Any]] = Field(
        description="Its columns in order: name, label, unit, and for coded text the "
                    "label of each value.")


class UnavailableSpec(BaseModel):
    """One reason a statistic can be unavailable."""
    code: str = Field(description="The code a block's `unavailable` gives.")
    label: str = Field(description="A short name for it.")
    help: str = Field(description="One line saying what is missing.")


class AnalysisCatalogue(BaseModel):
    """Response of `GET /analysis/options`: everything an analysis can be
    asked for, so a client hard-codes none of it."""
    settings: list[dict[str, Any]] = Field(
        description="Each section of settings in order, each with its parameters in the "
                    "shape `GET /backtest/options` describes a backtest's.")
    statistics: list[StatisticSpec] = Field(description="Every statistic, in order.")
    reports: list[ReportSpec] = Field(description="Every report, in order.")
    names: dict[str, dict[str, str]] = Field(
        description="Labels for figure names that are not statistic ids, by dimension: "
                    "books, methods, confidence levels, horizons, comparators, fields.")
    tables: list[TableSpec] = Field(description="The large tables a run gives.")
    unavailable: list[UnavailableSpec] = Field(description="Every reason a statistic can "
                                                           "be unavailable.")
    renamed: dict[str, str] = Field(description="Statistic ids renamed in this release, "
                                                "old to new.")
    dropped: list[str] = Field(description="Statistic ids no longer given.")
