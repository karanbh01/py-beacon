# src/beacon/analysis/reports/rerun.py
"""
A stored run's analysis asked for again: with statistics added, a time view
changed, or other choices.

    block = rerun(run, request, source, folder, previous=stored_block)

With the same choices as the block it is given, only the statistics that
block lacks (or lacks in a view now wanted) are worked out, and the answer is
the whole block, the earlier statistics and reports with the new ones. With
other choices everything asked for is worked out afresh. The run's factor
work is read from its folder when it was kept for these factors, and kept
there when it is worked out.
"""
# BN-399. Agreed with beacon-ui on 2026-10-29: a rerun answers the whole
# merged block, so the app replaces one object.
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ...data.fetcher import DataFetcher
from ..data import RunData
from ..request import AnalysisRequest, analyse_run
from .context import Context


def rerun(run: RunData,
          request: AnalysisRequest,
          source: DataFetcher | None,
          folder: Path | None = None,
          previous: Mapping[str, Any] | None = None,
          progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """The analysis *request* asks for of a stored run.

    Args:
        run: The run.
        request: What to give.
        source: The data store, which factor, liquidity and risk figures
            read; None when it is not loaded.
        folder: Where the run is kept, which its factor work is read from
            and kept in.
        previous: The run's block so far, whose statistics are kept when it
            was made with the same choices.
        progress: Told as each part starts.

    Returns:
        dict: The block, as `beacon.analysis.request.analyse` gives it.
    """
    context = Context(run, request.options, source, request.factors, folder)
    kept = reusable(previous, request)
    # What the run could not give before, it cannot give now either.
    lacking = ({statistic_id for statistic_id in request.wanted()
                if statistic_id in previous.get("unavailable", {})}
               if previous is not None and same_choices(previous, request) else set())
    block = analyse_run(context, request, progress, skip=set(kept) | lacking)

    if folder is not None:
        context.keep_work(folder)

    if previous is None or not same_choices(previous, request):
        return block

    statistics = {**previous.get("statistics", {}), **kept, **block["statistics"]}
    unavailable = {statistic_id: reason for statistic_id, reason in {
        **previous.get("unavailable", {}), **block["unavailable"]}.items()
        if statistic_id not in statistics}

    return {**block, "statistics": statistics, "unavailable": unavailable,
            "reports": {**previous.get("reports", {}), **block["reports"]},
            "groupings": block.get("groupings", previous.get("groupings", []))}


def same_choices(previous: Mapping[str, Any],
                 request: AnalysisRequest) -> bool:
    """Whether *previous* was made with the choices *request* makes: the
    same options and factors."""
    asked = request.as_data()
    made = previous.get("request", {})

    return bool(made.get("options") == asked["options"]
                and made.get("factors") == asked["factors"])


def reusable(previous: Mapping[str, Any] | None,
             request: AnalysisRequest) -> dict[str, Any]:
    """The statistics of *previous* that *request* wants and need not be
    worked out again: made with the same choices, in every view now
    wanted."""
    if previous is None or not same_choices(previous, request):
        return {}

    found = previous.get("statistics", {})

    return {statistic_id: found[statistic_id]
            for statistic_id, views in request.wanted().items()
            if statistic_id in found
            and ("rolling" not in views or "rolling" in found[statistic_id])}
