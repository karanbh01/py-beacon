# src/beacon/synthetic/__init__.py
"""
Synthetic market data at demo scale.

Generates a universe of anonymised companies with prices that reproduce the
stylized facts of equity returns — volatility clustering, fat tails, negative
skew, and a factor structure that makes names co-move — together with the
reference data, shares outstanding, free float and corporate actions that go
with them.

    from beacon.synthetic import SyntheticConfig, generate

    dataset = generate(SyntheticConfig(assets=64, seed=1))
    fetcher = dataset.fetcher()

Or from the command line, writing a store the server auto-loads:

    python -m beacon.synthetic --assets 512 --start 2019-12-31 --seed 42

Nothing generated here resembles a real company: names are ``Company A`` …
and every ticker carries a ``CMP`` prefix, which makes a collision with a real
listing impossible rather than merely improbable.

This is **not** `beacon.testing.dataset`, which is a tiny frozen fixture whose
exact values chart baselines depend on. See `dataset.py` for why the two must
stay apart.
"""
from .dataset import (
    DEFAULT_ASSETS,
    DEFAULT_END,
    DEFAULT_SEED,
    DEFAULT_START,
    SyntheticConfig,
    SyntheticDataset,
    generate,
    write,
)

__all__ = [
    "DEFAULT_ASSETS",
    "DEFAULT_END",
    "DEFAULT_SEED",
    "DEFAULT_START",
    "SyntheticConfig",
    "SyntheticDataset",
    "generate",
    "write",
]
