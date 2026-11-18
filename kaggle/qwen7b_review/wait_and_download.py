"""Wait for the private Qwen 7B ANSG review kernel and collect its outputs."""
import json
import subprocess
import sys
import time
from pathlib import Path

REF = "mossabibrahim/ansg-qwen-7b-plan-story-review"
OUTPUT = Path("kaggle-qwen7b-output")
DEADLINE = time.monotonic() + 7800


def cli(*args):
    return subprocess.run(["kaggle", "kernels", *args], capture_output=True, text=True)


while time.monotonic() < DEADLINE:
    response = cli("status", REF)
    line = (response.stdout + "\n" + response.stderr).strip()
    if response.returncode:
        print("Kaggle status error:", line[-800:], flush=True)
        if "kernels.get" in line and "denied" in line:
            listing = cli("list", "--mine", "--page-size", "50")
            print("Kaggle kernels owned by this account:", listing.stdout[-4000:], flush=True)
            raise SystemExit("Kaggle denied status access; inspect the actual kernel ref above.")
    else:
        state = next((x for x in ["COMPLETE", "ERROR", "CANCELLED", "RUNNING", "QUEUED"]
                      if "KernelWorkerStatus." + x in line), None)
        print("Kaggle Qwen 7B review:", state or line[-150:], flush=True)
        if state in ("COMPLETE", "ERROR", "CANCELLED"):
            OUTPUT.mkdir(parents=True, exist_ok=True)
            result = cli("output", REF, "-p", str(OUTPUT))
            if result.returncode:
                print("Output download failed:", result.stderr[-800:], flush=True)
                raise SystemExit(1)
            reports = list(OUTPUT.rglob("model_comparison.json"))
            if state != "COMPLETE" or not reports:
                raise SystemExit("Kaggle generation failed or report missing.")
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            if report.get("status") != "completed" or set(report.get("conditions", {})) != {
                "direct", "ansg_plan", "ansg_plan_repaired"
            }:
                raise SystemExit("Generated report incomplete.")
            print("Verified real GPU inference:", json.dumps({
                "model": report.get("model"),
                "gpus": report.get("gpu_names"),
                "plan_validation": report.get("plan_validation"),
                "word_counts": {
                    k: v.get("word_count_whitespace")
                    for k, v in report.get("conditions", {}).items()
                },
            }, ensure_ascii=False), flush=True)
            raise SystemExit(0)
    time.sleep(40)
raise SystemExit("Timed out waiting for Kaggle generation.")
