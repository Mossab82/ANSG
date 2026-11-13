"""Minimal ANSG Kaggle automation connectivity test; no training."""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

try:
    import torch
    gpu_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
except ImportError:
    gpu_count = 0
    gpu_names = []

report = {
    "test": "ansg_kaggle_api_smoke",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "gpu_count": gpu_count,
    "gpu_names": gpu_names,
    "passed": gpu_count >= 1,
    "training_performed": False,
}
output = Path("/kaggle/working/ansg_smoke_report.json")
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
if not report["passed"]:
    raise RuntimeError("No GPU detected; check Kaggle accelerator settings.")
