# src/beacon/backtest/presets.py
"""
Vehicles for real fund structures, ready to use.

    Backtest(initial_capital=1e8, vehicle=uk_oeic(management_fee_bps=15))
    Backtest(initial_capital=1e8, vehicle=preset("luxembourg_sicav"))

A preset is a `Vehicle` with the settings a structure usually has: how it
prices dealing and the diversification limits it must keep. Every setting
can be changed by passing it, so a preset is a starting point, not a rule:

    uk_oeic(pricing=DilutionLevy(threshold=0.05))

| Preset | Pricing | Diversification limits |
| --- | --- | --- |
| `uk_oeic` | Full swing pricing | UCITS |
| `luxembourg_sicav` | Partial swing pricing, above 2% of the fund | UCITS |
| `irish_icav` | Anti-dilution levy, above 2% of the fund | UCITS |
| `us_mutual_fund` | Single pricing, with an optional redemption fee | 1940 Act |

Swing factors and levies are estimated from the run's cost model unless set.
The UCITS limits are those for a fund replicating an index (20% in one
issuer, 35% in the largest), since a backtest tracks an index.
"""
# BN-268, phase 6 of decisions/0006. The ETF presets join with the ETF
# vehicle. See docs/concepts/fund-vehicles.md for the reasoning behind each.
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .dealing import OUT, DilutionLevy, SwingPricing
from .limits import Act1940Limits, UcitsLimits
from .vehicle import Vehicle

# The threshold above which a partial swing or a levy applies, as a share of
# the fund: a common choice, and only a default.
LARGE_DEAL = 0.02


def uk_oeic(**settings: Any) -> Vehicle:
    """A UK open-ended investment company: daily dealing, a full swinging
    single price, and the UCITS limits."""
    return _built("UK OEIC", settings, pricing=SwingPricing(),
                  limits=[UcitsLimits()])


def luxembourg_sicav(**settings: Any) -> Vehicle:
    """A Luxembourg SICAV under UCITS: a partial swing above 2% of the fund,
    and the UCITS limits."""
    return _built("Luxembourg SICAV", settings,
                  pricing=SwingPricing(threshold=LARGE_DEAL),
                  limits=[UcitsLimits()])


def irish_icav(**settings: Any) -> Vehicle:
    """An Irish ICAV under UCITS: an anti-dilution levy on deals above 2% of
    the fund, and the UCITS limits."""
    return _built("Irish ICAV", settings,
                  pricing=DilutionLevy(threshold=LARGE_DEAL),
                  limits=[UcitsLimits()])


def us_mutual_fund(redemption_fee_bps: float = 0.0,
                   **settings: Any) -> Vehicle:
    """A US mutual fund: single pricing, an optional redemption fee, and the
    1940 Act's test for a diversified fund.

    Args:
        redemption_fee_bps: A fee on redemptions, paid into the fund, in
            basis points. Charged on every redemption, since a run does not
            track how long each investor held.
        **settings: Any other `Vehicle` setting.
    """
    pricing = (DilutionLevy(rate_bps=redemption_fee_bps, on=OUT)
               if redemption_fee_bps > 0 else None)

    return _built("US mutual fund", settings, pricing=pricing,
                  limits=[Act1940Limits()])


@dataclass(frozen=True)
class Preset:
    """A preset by name, for listing.

    Attributes:
        key: The name `preset()` takes.
        name: What the structure is called.
        build: Makes the vehicle; takes any `Vehicle` setting to change.
    """
    key: str
    name: str
    build: Callable[..., Vehicle]


PRESETS: dict[str, Preset] = {
    preset.key: preset for preset in (
        Preset("uk_oeic", "UK OEIC", uk_oeic),
        Preset("luxembourg_sicav", "Luxembourg SICAV", luxembourg_sicav),
        Preset("irish_icav", "Irish ICAV", irish_icav),
        Preset("us_mutual_fund", "US mutual fund", us_mutual_fund),
    )
}


def preset(key: str,
           **settings: Any) -> Vehicle:
    """The vehicle for a preset, by name, with any settings changed.

    Raises:
        KeyError: If there is no preset by that name; the message lists them.
    """
    if key not in PRESETS:
        raise KeyError(f"No preset {key!r}. Presets: {', '.join(PRESETS)}.")

    return PRESETS[key].build(**settings)


def _built(name: str,
           settings: dict[str, Any],
           **defaults: Any) -> Vehicle:
    """A vehicle named *name*: *defaults*, with *settings* laid over them."""
    chosen = {key: value for key, value in defaults.items() if value is not None}
    chosen.update(settings)
    chosen.setdefault("name", name)

    return Vehicle(**chosen)
