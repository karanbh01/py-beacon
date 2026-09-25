# src/beacon/index/requirements.py
"""
Whether a dataset has the columns a definition needs, asked once up front.

Every rule and weighting scheme declares the market columns it reads
(`required_columns`), and `require_columns` checks the union, plus the price
column the calculation values holdings with, against the dataset **before a
run does any work**. A missing column is refused with a `CalculationError`
that names each column, what needs it, and what the dataset has instead.

Without this check, a run would get as far as the first read that needed the
column and then fail in terms of one company on one day (for example, "N0 has
no positive SHARES_OUTSTANDING on 2024-01-02"), sending the reader to inspect
that company's data when the dataset has no share-count column at all. A
missing VOLUME column is worse: every name fails a liquidity screen, and the
run fails as "index holds nothing on its base date" with no mention of volume.
A column cannot appear or vanish halfway through a run, so asking once is
enough.

When the data provider cannot say which columns it has (it has no
`market_columns` list), the check is skipped and the run fails, if it fails,
at the first read.
"""
# Added in BN-217, after the two misleading failures described above.
#
# What this does not replace: `DataFetcher._market_scalar` still checks for a
# column before reading the frame, because it serves callers that have no
# definition to validate against (the reference endpoint, and anyone using the
# fetcher directly). Those questions arrive with nothing declared in advance,
# so the lookup answers "no value" for a column that is not there rather than
# raising.
from collections.abc import Iterable

from ..data.fetcher import DataFetcher
from ..exceptions import CalculationError
from .constructor import IndexDefinition


def required_by(definition: IndexDefinition,
                price_column: str) -> dict[str, list[str]]:
    """Every market column a definition reads, and what reads it.

    Args:
        definition: The index definition to check.
        price_column: The column the calculation values holdings with every
            day. It is a requirement of the calculation itself, separate from
            any rule or scheme.

    Returns:
        dict: column -> the parts of the definition that need it, in the order
        they appear. The *who* is what makes the refusal actionable: "needs
        SHARES_OUTSTANDING" says what to load, and "for MarketCapWeighted" says
        what to change instead if loading it is not an option.
    """
    needs: dict[str, list[str]] = {}

    def note(columns: Iterable[str],
             who: str) -> None:
        for column in sorted(columns):
            needs.setdefault(column, [])

            if who not in needs[column]:
                needs[column].append(who)

    note([price_column], "the index calculation, to value holdings daily")
    note(definition.weighting_scheme.required_columns(),
         _describe_scheme(definition.weighting_scheme))

    for rule in definition.eligibility_rules:
        note(rule.required_columns(), rule.rule_name)

    return needs


def require_columns(definition: IndexDefinition,
                    fetcher: DataFetcher,
                    price_column: str) -> None:
    """Refuse a definition the dataset cannot support, before any work is done.

    Args:
        definition: The index definition about to run.
        fetcher: The data it would run on.
        price_column: The calculation's daily valuation column.

    Raises:
        CalculationError: If any column the definition reads is absent from the
            dataset's market data, naming each one, what needs it, and what the
            dataset has instead.
    """
    available = _available_columns(fetcher)

    if available is None:
        return

    missing = {column: who
               for column, who in required_by(definition, price_column).items()
               if column not in available}

    if not missing:
        return

    raise CalculationError(
        calculation_name="DataRequirements",
        details=_describe_missing(definition, missing, sorted(available)))


def require_price_column(fetcher: DataFetcher,
                         price_column: str,
                         who: str) -> None:
    """The one requirement a run with no definition still has: a price.

    Used by the backtest engine, which may be driven by a raw weight schedule
    with no definition behind it, and so has nothing to declare beyond the
    column it marks positions at. Skipped when the provider cannot list its
    columns.

    Args:
        fetcher: The data the run would use.
        price_column: The column positions are priced from.
        who: What needs the column, for the error message.

    Raises:
        CalculationError: If the dataset has no such column.
    """
    available = _available_columns(fetcher)

    if available is None or price_column in available:
        return

    raise CalculationError(
        calculation_name="DataRequirements",
        details=(f"{who} prices positions from the {price_column} column, and "
                 f"the dataset's market data has no such column -- it has "
                 f"{', '.join(sorted(available)) or 'no columns at all'}. Load "
                 f"{price_column}, or name a price column the dataset carries."))


def _available_columns(fetcher: DataFetcher) -> set[str] | None:
    """The dataset's market columns, or None when the provider cannot say.

    The provider is an interface rather than a class -- the engine already
    treats `warm_session` and `delisting_dates` as capabilities a
    hand-assembled provider may lack -- and a check that cannot learn what is
    available has nothing to check against. Skipping is the honest answer: the
    run then fails where it always did, at the first read, with the message it
    always had, and loses nothing it previously had.

    A `MagicMock` is the case that forced this. Asked for `market_columns` it
    returns a Mock, iterating a Mock yields nothing, and the check read that as
    a dataset with no columns at all -- refusing every mocked run for want of
    a CLOSE column that the mock was about to supply.
    """
    columns = getattr(fetcher, "market_columns", None)

    if not isinstance(columns, (list, tuple)):
        return None

    return {str(column) for column in columns}


def _describe_scheme(scheme: object) -> str:
    """A scheme's name, with the input that changed what it reads."""
    name = str(getattr(scheme, "scheme_name", type(scheme).__name__))

    if getattr(scheme, "use_free_float", False):
        return f"{name} (free-float adjusted)"

    return name


def _describe_missing(definition: IndexDefinition,
                      missing: dict[str, list[str]],
                      available: list[str]) -> str:
    """The refusal, stated as the dataset's problem rather than a company's."""
    lines = "; ".join(f"{column}, for {' and '.join(who)}"
                      for column, who in sorted(missing.items()))

    return (f"index '{definition.index_id}' cannot run on this dataset. It needs "
            f"{lines}, and the dataset's market data has no such "
            f"column{'s' if len(missing) > 1 else ''} -- it has "
            f"{', '.join(available) or 'no columns at all'}. This is a property "
            f"of the whole dataset, not of any one company. Load the missing "
            f"column{'s' if len(missing) > 1 else ''}, or change the definition "
            f"so it does not need {'them' if len(missing) > 1 else 'it'}.")
