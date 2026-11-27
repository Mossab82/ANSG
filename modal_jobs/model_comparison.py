"""Run the Arabic model comparison on Modal (A10G GPU) instead of Kaggle.

Reuses kaggle/model_comparison/run_model_comparison.py unchanged: the output
directory and the runtime pip step are overridden here. Inference only; no
dissertation result is reproduced.

Usage (from the repository root):  modal run modal_jobs/model_comparison.py
"""
from pathlib import Path

import modal

app = modal.App("ansg-model-comparison")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.46,<5",
        "accelerate>=1.0",
        "sentencepiece",
        "protobuf",
    )
    .add_local_file(
        "kaggle/model_comparison/run_model_comparison.py", "/root/run_model_comparison.py"
    )
)


@app.function(image=image, gpu="A10G", timeout=3 * 3600)
def compare() -> dict[str, str]:
    import sys

    sys.path.insert(0, "/root")
    import run_model_comparison as job

    job.OUT = Path("/tmp/ansg_model_comparison")
    job.sh = lambda *args: None  # dependencies are already in the image
    job.main()
    return {p.name: p.read_text(encoding="utf-8") for p in job.OUT.iterdir() if p.is_file()}


@app.local_entrypoint()
def main(out_dir: str = "modal-model-comparison-output"):
    files = compare.remote()
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (target / name).write_text(text, encoding="utf-8")
    print("wrote", sorted(files))
