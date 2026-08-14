# scripts/fuzz_store.py
"""
Write the store the nightly fuzz run serves.

    python scripts/fuzz_store.py .fuzzstore

The canonical frozen fixture rather than the generator: five names and a few
hundred days is plenty to exercise a handler, and it keeps the step to about a
second. The point of the run is to throw unusual *input* at the endpoints, not
to give them a lot of data to chew on.

## Why this exists at all

It replaces `scripts/fuzz_app.py`, which built the application in-process for
`schemathesis run --app`. That option was removed in schemathesis v4, so the
server is now started for real and fuzzed over a socket -- which needs a store
on disk rather than a fetcher in memory.

The token the workflow uses is fixed and passed on the command line. That is
not a credential: the process is short-lived, bound to loopback, and holds
nothing but synthetic data. Authentication has its own contract test, and the
fuzz run excludes the `ignored_auth` check for the same reason -- it is there
to exercise handlers, not to rediscover that they are guarded.
"""
import sys
from pathlib import Path

from beacon.data import store
from beacon.testing import dataset


def main(argv: list[str] | None = None) -> int:
    """Write the fixture to a store directory.

    Args:
        argv: Argument list, defaulting to sys.argv[1:]. One positional
            argument: where to write.

    Returns:
        int: Process exit code. 2 if the destination was not given.
    """
    arguments = sys.argv[1:] if argv is None else argv

    if len(arguments) != 1:
        print(f"usage: {Path(__file__).name} <store-directory>",
              file=sys.stderr)

        return 2

    written = store.save(dataset.data_fetcher(), Path(arguments[0]),
                         source=store.SOURCE_SYNTHETIC)

    print(f"Wrote the fuzz store to {written}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
