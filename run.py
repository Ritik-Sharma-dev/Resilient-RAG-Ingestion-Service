"""Single-command launcher: starts the FastAPI backend and Streamlit
frontend together, and shuts both down on Ctrl+C."""

import signal
import subprocess
import sys
import time

processes: list[subprocess.Popen] = []


def shutdown(*_args):
    print("\nStopping backend and frontend...")
    for proc in processes:
        proc.terminate()
    for proc in processes:
        proc.wait()
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("Starting backend (FastAPI) on http://127.0.0.1:8000 ...")
    backend = subprocess.Popen([sys.executable, "-m", "app.main"])
    processes.append(backend)

    time.sleep(3)  # give the backend a head start before the frontend tries to reach it

    print("Starting frontend (Streamlit) on http://127.0.0.1:8501 ...")
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/app.py"]
    )
    processes.append(frontend)

    print("\nBoth running. Press Ctrl+C to stop both.\n")

    while True:
        for proc in processes:
            code = proc.poll()
            if code is not None:
                print(f"Process {proc.args} exited with code {code} — stopping the other one.")
                shutdown()
        time.sleep(1)


if __name__ == "__main__":
    main()
