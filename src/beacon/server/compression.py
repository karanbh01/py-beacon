# src/beacon/server/compression.py
"""
Compressing large responses for a client on another machine, and not for one
on the same machine, where compressing costs more time than the transfer it
saves.
"""
# BN-398. A 6,000-name run's trades are 8.2 MB as Arrow and 2.0 MB gzipped;
# gzipping them takes about 165 ms, and moving 8 MB on the same machine a few.
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

# Responses at least this large are compressed.
MINIMUM_SIZE = 1_000_000

# The hosts a request from the same machine comes from.
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})


class RemoteGZip:
    """Gzip responses of at least `MINIMUM_SIZE` bytes, for clients on
    another machine only."""

    def __init__(self,
                 app: ASGIApp,
                 minimum_size: int = MINIMUM_SIZE):
        self.app = app
        self.compressing = GZipMiddleware(app, minimum_size=minimum_size)

    async def __call__(self,
                       scope: Scope,
                       receive: Receive,
                       send: Send) -> None:
        client = scope.get("client")
        local = scope["type"] != "http" or client is None or client[0] in LOOPBACK

        await (self.app if local else self.compressing)(scope, receive, send)
