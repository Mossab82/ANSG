"""Wait for a private Kaggle smoke kernel to finish and validate its output.

GitHub Actions reads its token from repository Actions Secrets.
No token is written to artifacts or printed by this script.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REF = "mossabibrahim/ansg-api-smoke"
OUTPUT = Path("kaggle-output")
MAX_WAIT_SECONDS = 780
POLL_SECONDS = 20

def kaggle(*arguments):
    return subprocess.run(
        ["kaggle", "kernels", *arguments],
        text=True, capture_output=True, check=False
    )

def main():
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        response = kaggle("status", REF)
        status_text = (response.stdout + "\n" + response.stderr).strip()
        if response.returncode != 0:
            print("Kaggle status temporarily unavailable; retrying.")
        else:
            # The Kaggle CLI returns KernelWorkerStatus.RUNNING, etc.
            for state in ("COMPLETE", "ERROR", "CANCELLED", "RUNNING", "QUEUED"):
                if "KernelWorkerStatus." + state in status_text:
                    print("Kaggle status:", state, flush=True)
                    if state in ("ERROR", "CANCELLED"):
                        raise RuntimeError("Kaggle kernel ended with " + state)
                    if state == "COMPLETE":
                        OUTPUT.mkdir(parents=True, exist_ok=True)
                        result = kaggle("output", REF, "-p", str(OUTPUT))
                        if result.returncode:
                            raise RuntimeError(
                                "Output download failed: " + result.stderr[-900:]
                            )
                        report_file = OUTPUT / "ansg_smoke_report.json"
                        if not report_file.is_file():
                            raise RuntimeError(
                                "Kernel completed but expected smoke report is missing."
                            )
                        report = json.loads(report_file.read_text(encoding="utf-8"))
                        if report.get("passed") is not True or report.get("gpu_count", 0) < 1:
                            raise RuntimeError("GPU validation failed; inspect smoke report.")
                        print("Smoke test passed. GPUs:", report["gpu_count"])
                        print("GPU names:", report.get("gpu_names", []))
                        return
                    break
            else:
                print("Unrecognized Kaggle status; retrying.")
        time.sleep(POLL_SECONDS)
    raise TimeoutError("Kaggle did not finish within the polling window.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("::error::" + str(exc), file=sys.stderr)
        raise SystemExit(1)
