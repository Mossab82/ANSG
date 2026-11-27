"""Wait for the private Kaggle model-comparison kernel and collect its outputs."""
import json
import subprocess
import time
from pathlib import Path

REF = "mossabibrahim/ansg-arabic-model-comparison"
OUTPUT = Path("kaggle-model-comparison-output")
DEADLINE = time.monotonic() + 5.5 * 3600


def cli(*args):
    return subprocess.run(["kaggle", "kernels", *args], capture_output=True, text=True)


while time.monotonic() < DEADLINE:
    response = cli("status", REF)
    line = (response.stdout + "\n" + response.stderr).strip()
    if response.returncode:
        print("Kaggle status error:", line[-800:], flush=True)
    else:
        state = next((x for x in ["COMPLETE", "ERROR", "CANCELLED", "RUNNING", "QUEUED"]
                      if "KernelWorkerStatus." + x in line), None)
        print("Kaggle model comparison:", state or line[-150:], flush=True)
        if state in ("COMPLETE", "ERROR", "CANCELLED"):
            OUTPUT.mkdir(parents=True, exist_ok=True)
            result = cli("output", REF, "-p", str(OUTPUT))
            print(result.stdout[-2000:], result.stderr[-2000:], flush=True)
            reports = list(OUTPUT.rglob("model_comparison.json"))
            if not reports:
                raise SystemExit(f"Kernel ended with {state}; no report was produced.")
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            print(json.dumps({r["model"]: r["status"] for r in report.get("results", [])}), flush=True)
            raise SystemExit(0 if report.get("status") == "completed" else 1)
    time.sleep(60)
raise SystemExit("Timed out waiting for Kaggle model comparison.")
