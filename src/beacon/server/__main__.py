# src/beacon/server/__main__.py
"""
Launcher for the Beacon API server.

    python -m beacon.server --port 0

The desktop client spawns this process and needs to know where it landed. With
``--port 0`` the OS picks a free port, so the socket is bound here — before
uvicorn starts — and the resulting port is printed on stdout as
``BEACON_PORT=<n>`` and flushed. The client can read that line, then treat
every later stdout line as ordinary logging.
"""
import argparse
import socket
import sys

from .._optional import require
from .app import create_app
from .config import TOKEN_ENV_VAR, ServerConfig

require("uvicorn", "The Beacon API server")

import uvicorn  # noqa: E402

# Printed verbatim so the client can match on it rather than parse free text.
PORT_ANNOUNCEMENT = "BEACON_PORT="


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: Parser for --host, --port and --token.
    """
    parser = argparse.ArgumentParser(
        prog="python -m beacon.server",
        description="Run the local Beacon API server.")

    parser.add_argument("--host",
                        default="127.0.0.1",
                        help="interface to bind (default: 127.0.0.1, loopback only)")
    parser.add_argument("--port",
                        type=int,
                        default=0,
                        help="port to bind; 0 lets the OS choose (default: 0)")
    parser.add_argument("--token",
                        default=None,
                        help=f"bearer token; falls back to ${TOKEN_ENV_VAR}")

    return parser


def bind_socket(host: str,
                port: int) -> socket.socket:
    """Bind a listening socket and return it.

    Binding here rather than inside uvicorn is what makes ``--port 0`` usable:
    the OS assigns the port at bind time, so it can be reported before the
    server begins serving.

    Args:
        host: Interface to bind.
        port: Port to bind; 0 asks the OS for a free one.

    Returns:
        socket.socket: The bound, listening socket.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen()

    return sock


def announce_port(port: int) -> None:
    """Print the bound port on stdout and flush it.

    The flush matters: the parent process blocks on this line, and a buffered
    stdout would deadlock the handshake.

    Args:
        port: The port actually bound.
    """
    print(f"{PORT_ANNOUNCEMENT}{port}", flush=True)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, bind, announce the port, and serve.

    Args:
        argv: Argument list, defaulting to sys.argv[1:].

    Returns:
        int: Process exit code. 2 if configuration is invalid.
    """
    args = build_parser().parse_args(argv)

    try:
        config = ServerConfig.from_environment(token=args.token,
                                               host=args.host,
                                               port=args.port)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    sock = bind_socket(config.host, config.port)
    announce_port(sock.getsockname()[1])

    app = create_app(config)
    server = uvicorn.Server(uvicorn.Config(app, log_level="info"))

    try:
        server.run(sockets=[sock])
    finally:
        sock.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
