"""Poll Kaggle heritage pilot and download independent reports and checkpoints."""
import json
import subprocess
import sys
import time
from pathlib import Path

REF = "mossabibrahim/ansg-heritage-lora-pilot"
OUTPUT = Path("kaggle-heritage-output")
DEADLINE = time.monotonic() + 7800

def cli(*args):
    return subprocess.run(["kaggle", "kernels", *args], capture_output=True, text=True)

while time.monotonic() < DEADLINE:
    response=cli("status",REF)
    line=(response.stdout+"\n"+response.stderr).strip()
    if response.returncode:
        print("Kaggle status unavailable, retrying.",flush=True)
    else:
        state=next((x for x in ["COMPLETE","ERROR","CANCELLED","RUNNING","QUEUED"]
                    if "KernelWorkerStatus."+x in line),None)
        print("Kaggle pilot:",state or line[-150:],flush=True)
        if state in ["COMPLETE","ERROR","CANCELLED"]:
            OUTPUT.mkdir(parents=True,exist_ok=True)
            result=cli("output",REF,"-p",str(OUTPUT))
            if result.returncode:
                print("Output download failed:",result.stderr[-500:],flush=True)
                raise SystemExit(1)
            # Kaggle kernel output command sometimes nests results in one directory.
            final=list(OUTPUT.rglob("final_report.json"))
            failures=list(OUTPUT.rglob("failure_report.json"))
            if failures:
                print("Pilot failed:",failures[0].read_text(encoding="utf-8")[-1500:])
            if state!="COMPLETE" or not final:
                raise SystemExit("Kaggle pilot failed or did not save final_report.json.")
            result=json.loads(final[0].read_text(encoding="utf-8"))
            if result.get("status")!="completed" or not result.get("steps_finished"):
                raise SystemExit("Invalid training result; no completed optimizer steps.")
            if not list(OUTPUT.rglob("adapter_final")):
                raise SystemExit("Adapter checkpoint directory missing.")
            print("Verified real heritage adaptation pilot:",json.dumps({
                k:result.get(k) for k in ("steps_finished","baseline_heldout_loss",
                   "best_heldout_loss","last_heldout_loss","trained_CSAM_DAL")
            }),flush=True)
            raise SystemExit(0)
    time.sleep(40)
raise SystemExit("Timeout: Kaggle pilot did not complete within 130 minutes.")
