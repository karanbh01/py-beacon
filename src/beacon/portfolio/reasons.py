# src/beacon/portfolio/reasons.py
"""
Why a trade was made.

Each trade has one reason, the event that triggered it, and a set of tags:
the reason, what it implies, and anything else that shaped it. On a
rebalance the reason is the rebalance, and the tags name the stages that
changed the name's target (an issuer cap, a liquidity screen). A trade made
to move cash in or out of the book is tagged a cash raise when it sells and a
cash spend-down when it buys, whatever moved the cash.

Figures by reason add up to the run's; figures by tag may count a trade more
than once, as one trade can carry several.
"""
# BN-406. The reasons as agreed with Karan on 2026-10-27. Index change, drift,
# discretionary and risk-driven have no engine feature yet: their names are
# kept for when they do.

# Index.
REBALANCE = "rebalance"
INDEX_CHANGE = "index change"
DRIFT = "drift"

# Limits: tags a rebalance trade carries when a stage changed its target.
DIVERSIFICATION = "diversification"
POSITION_LIMIT = "position limit"
LIQUIDITY = "liquidity"
EXCLUSION = "exclusion"

# Strategy.
MODEL_DRIVEN = "model-driven"
DISCRETIONARY = "discretionary"
RISK_DRIVEN = "risk-driven"

# Flows.
SUBSCRIPTION = "subscription"
REDEMPTION = "redemption"

# Cash.
CASH_RAISE = "cash raise"
CASH_SPEND_DOWN = "cash spend-down"
DIVIDEND_REINVESTMENT = "dividend reinvestment"
DISTRIBUTION_FUNDING = "distribution funding"
FEE_FUNDING = "fee funding"
SETTLEMENT = "settlement"

# Corporate actions and execution.
CORPORATE_ACTION = "corporate action"
COMPLETION = "completion"

# Derivatives: their trades are recorded with the derivatives (BN-409).
FUTURES_ROLL = "futures roll"
FUTURES_EXPOSURE = "futures exposure"
FX_HEDGE = "fx hedge"

# Each reason's category, for grouping them in a report.
CATEGORIES = {
    REBALANCE: "index", INDEX_CHANGE: "index", DRIFT: "index",
    DIVERSIFICATION: "limits", POSITION_LIMIT: "limits", LIQUIDITY: "limits",
    EXCLUSION: "limits",
    MODEL_DRIVEN: "strategy", DISCRETIONARY: "strategy", RISK_DRIVEN: "strategy",
    SUBSCRIPTION: "flows", REDEMPTION: "flows",
    CASH_RAISE: "cash", CASH_SPEND_DOWN: "cash", DIVIDEND_REINVESTMENT: "cash",
    DISTRIBUTION_FUNDING: "cash", FEE_FUNDING: "cash", SETTLEMENT: "cash",
    CORPORATE_ACTION: "corporate actions", COMPLETION: "execution",
    FUTURES_ROLL: "derivatives", FUTURES_EXPOSURE: "derivatives", FX_HEDGE: "derivatives",
}

REASONS = tuple(CATEGORIES)

# Reasons whose trades move cash into or out of the book: a sell raises cash,
# a buy spends it down.
MOVES_CASH = frozenset({SUBSCRIPTION, REDEMPTION, CASH_RAISE, CASH_SPEND_DOWN,
                        DIVIDEND_REINVESTMENT, DISTRIBUTION_FUNDING, FEE_FUNDING, SETTLEMENT})


def tags_for(reason: str,
             side: str,
             extra: tuple[str, ...] = ()) -> tuple[str, ...]:
    """A trade's tags: its reason, a cash raise or spend-down when its reason
    moves cash, and *extra*, each once, in that order."""
    found = [reason] if reason else []

    if reason in MOVES_CASH:
        found.append(CASH_RAISE if side.upper() == "SELL" else CASH_SPEND_DOWN)

    found.extend(extra)

    return tuple(dict.fromkeys(found))
