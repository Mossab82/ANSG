"""Issue #6 item 4: importing the data/knowledge layer must not require PyTorch,
open corpus/data files, or touch the network.

Each check runs in a fresh interpreter so that modules imported by other tests
cannot hide a side effect. A ``sys.addaudithook`` hook records ``socket.*``
events and every ``open`` of a non-Python file during the imports.
"""

import json
import subprocess
import sys
from pathlib import Path

import ansg

PKG_DIR = Path(ansg.__file__).resolve().parent

# Model layer (Issue #11): the only modules allowed to import PyTorch. Every other
# ansg module, including any added later, must import without torch.
MODEL_LAYER = {
    "ansg.csam", "ansg.dal", "ansg.model", "ansg.training", "ansg.pretrained",
    "ansg.provenance", "ansg.data.training_dataset",
}

PROBE = r"""
import json, pkgutil, sys, importlib
events = {"socket": [], "open": []}
def hook(event, args):
    if event.startswith("socket."):
        events["socket"].append(event)
    elif event == "open" and args and isinstance(args[0], str):
        path = args[0]
        if not path.endswith((".py", ".pyc", ".pth")) and "/__pycache__" not in path \
                and not path.startswith(("/proc", "/dev")) and "site-packages" not in path \
                and "/lib/python" not in path:
            events["open"].append(path)
sys.addaudithook(hook)
import ansg
MODEL_LAYER = set(json.loads(sys.argv[1]))
names = sorted(m.name for m in pkgutil.walk_packages(ansg.__path__, "ansg."))
for name in names:
    if name not in MODEL_LAYER:
        importlib.import_module(name)
print(json.dumps({"modules": names, "torch_imported": "torch" in sys.modules,
                  "transformers_imported": "transformers" in sys.modules, **events}))
"""


def _run_probe() -> dict:
    r = subprocess.run([sys.executable, "-c", PROBE, json.dumps(sorted(MODEL_LAYER))],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_every_module_imports_without_torch_network_or_data_files():
    result = _run_probe()
    expected = {"ansg.data", "ansg.data.nights", "ansg.data.records", "ansg.data.normalization",
                "ansg.knowledge.arabic_verbnet", "ansg.knowledge.arabic_namenet", "ansg.knowledge.ckg",
                "ansg.evaluation.craft", "ansg.evaluation.statistics", "ansg.utils.parameter_audit"}
    assert expected <= set(result["modules"])
    assert MODEL_LAYER <= set(result["modules"]), "MODEL_LAYER lists a module that no longer exists"
    # Backward-compatible lazy exports of the relocated model-branch modules exist
    # without importing torch (they are resolved only on attribute access).
    assert {"ansg.data", "ansg.evaluation", "ansg.knowledge"} <= set(result["modules"])
    assert result["torch_imported"] is False
    assert result["transformers_imported"] is False
    assert result["socket"] == []
    # Only the package's own source files may be opened while importing.
    unexpected = [p for p in result["open"] if not p.startswith(str(PKG_DIR))]
    assert unexpected == []


def test_probe_detects_a_network_call():
    """Negative control: the hook really sees socket activity."""
    code = PROBE.replace("import ansg\n", "import ansg, socket\nsocket.socket().close()\n", 1)
    r = subprocess.run([sys.executable, "-c", code, json.dumps(sorted(MODEL_LAYER))],
                       capture_output=True, text=True, check=True)
    assert json.loads(r.stdout.strip().splitlines()[-1])["socket"]
