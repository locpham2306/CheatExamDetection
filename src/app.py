"""Trình khởi chạy một nút cho giao diện Exam Cheat Detection."""

import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_APP = PROJECT_ROOT / "src" / "web.py"


def running_under_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except (ImportError, TypeError):
        return False


def find_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return server_socket.getsockname()[1]


def main():
    environment = os.environ.copy()
    environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    port = find_available_port()
    app_url = f"http://localhost:{port}"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(WEB_APP),
        "--server.headless=true",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
    ]
    process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment)

    try:
        for _ in range(60):
            if process.poll() is not None:
                return process.returncode
            try:
                with urllib.request.urlopen(
                    f"{app_url}/_stcore/health", timeout=0.5
                ) as response:
                    if response.status == 200:
                        if os.environ.get("EXAM_APP_NO_BROWSER") != "1":
                            webbrowser.open(app_url)
                        print(f"Exam Cheat Detection: {app_url}")
                        break
            except OSError:
                time.sleep(0.25)

        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        return process.wait(timeout=5)


if __name__ == "__main__":
    if running_under_streamlit():
        import streamlit as st

        st.error(
            "`app.py` là file khởi chạy. Hãy chạy `python src/app.py` hoặc "
            "chạy trực tiếp `streamlit run src/web.py`."
        )
    else:
        raise SystemExit(main())
