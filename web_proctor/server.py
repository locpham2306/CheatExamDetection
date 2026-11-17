"""Local server for the browser-based exam proctor.

The AI runtimes use SharedArrayBuffer, so the page must be served with
cross-origin isolation headers. A plain ``python -m http.server`` does not
add those headers.
"""

from __future__ import annotations

import argparse
import functools
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent


class ProctorRequestHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
        ".task": "application/octet-stream",
    }

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the web proctor locally")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the application in the default browser",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handler = functools.partial(ProctorRequestHandler, directory=str(APP_DIR))

    try:
        server = ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as exc:
        raise SystemExit(
            f"Cannot start web proctor on {args.host}:{args.port}: {exc}"
        ) from exc

    browser_host = "localhost" if args.host in {"127.0.0.1", "0.0.0.0"} else args.host
    url = f"http://{browser_host}:{args.port}"
    print(f"Web proctor: {url}")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web proctor...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
