"""Capture actual run metadata; never fabricate historical dates or checksums."""
import json
import platform
import hashlib
from datetime import datetime,timezone
from pathlib import Path
import torch
from . import __version__

def sha256_file(path):
    digest=hashlib.sha256()
    with open(path,'rb') as file:
        for chunk in iter(lambda:file.read(1<<20),b''): digest.update(chunk)
    return digest.hexdigest()

def write_run_manifest(destination,config,source_data=None,seed=42,extra=None):
    """`extra` (Issue #11): additional provenance fields, e.g. {'data_label': 'synthetic'}."""
    manifest={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
              'ansg_version':__version__,'python':platform.python_version(),
              'pytorch':torch.__version__,'cuda':torch.version.cuda,
              'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              'seed':seed,'configuration':config.__dict__,
              'source_data':{'path':str(source_data),'sha256':sha256_file(source_data)} if source_data else None,
              'historical_experiment':False,**dict(extra or {})}
    dest=Path(destination)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    return manifest
