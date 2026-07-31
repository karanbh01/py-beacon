# scripts/fuzz_app.py
"""
The application schemathesis fuzzes.

Built with the canonical dataset so endpoints that need data have some, and
with a temporary storage root so a fuzzing run cannot touch a real one. The
bearer token is fixed and passed on the command line, because the point of the
run is to exercise handlers rather than to rediscover that they are guarded —
authentication has its own contract test.

Loaded in-process by `schemathesis run --app`, so there is no server to start
and no port to race on, and a failure surfaces a Python traceback rather than a
connection error.
"""
import atexit
import tempfile
from pathlib import Path

from beacon.server import ServerConfig, create_app
from beacon.testing import dataset

# Fixed so the CI step can send it. Not a credential: the process is
# short-lived, local, and holds nothing but synthetic data.
TOKEN = "fuzz-token"

_storage = tempfile.TemporaryDirectory()
atexit.register(_storage.cleanup)

app = create_app(ServerConfig(auth_token=TOKEN,
                              data_fetcher=dataset.data_fetcher(),
                              storage_root=Path(_storage.name)))
