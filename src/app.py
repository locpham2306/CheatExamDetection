"""Trình khởi chạy một nút cho dashboard WebGPU Exam Cheat Detection."""

import os
import socket
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_SERVER = PROJECT_ROOT / "web_proctor" / "server.py"


def running_under_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except (ImportError, TypeError):
        return False


def find_available_port(preferred=8000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return server_socket.getsockname()[1]


def main():
    port = find_available_port()
    command = [
        sys.executable,
        str(WEB_SERVER),
        "--port",
        str(port),
    ]
    if os.environ.get("EXAM_APP_NO_BROWSER") == "1":
        command.append("--no-browser")
    process = subprocess.Popen(command, cwd=PROJECT_ROOT)

    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        return process.wait(timeout=5)


if __name__ == "__main__":
    if running_under_streamlit():
        import streamlit as st

        st.error(
            "`app.py` là launcher dashboard WebGPU. Hãy chạy bằng nút "
            "**Run Python File** hoặc lệnh `python src/app.py`, không chạy qua Streamlit."
        )
    else:
        raise SystemExit(main())
