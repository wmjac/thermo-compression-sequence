from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import numpy as np
import torch
from . import __version__
from .data import sha256_file


def json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False, default=json_default)
        handle.write("\n")
    os.replace(temporary, path)


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def source_manifest():
    root = Path(__file__).parent
    files = {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*.py"))}
    serialized = json.dumps(files, sort_keys=True).encode()
    return {"files": files, "sha256": hashlib.sha256(serialized).hexdigest()}


def collect_provenance(device):
    import scipy
    import matplotlib
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    selected_environment = ("CONDA_PREFIX", "LOADEDMODULES", "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK",
                            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    return {"created_at": timestamp(), "command": sys.argv, "python": sys.version,
            "python_executable": sys.executable, "platform": platform.platform(),
            "package_version": __version__, "versions": {"torch": torch.__version__, "numpy": np.__version__,
                "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
            "git_commit": commit, "git_dirty": dirty, "source": source_manifest(),
            "environment": {key: os.environ.get(key) for key in selected_environment},
            "device": str(device), "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_threads": torch.get_num_threads(),
            "torch_backend_settings": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_version": torch.backends.cudnn.version()},
            "physical_units": {"energy": "kT", "concentration": "nm^-3", "pressure": "kT nm^-3"}}


def snapshot_source(destination):
    root = Path(__file__).parent
    destination = Path(destination)
    for path in root.rglob("*.py"):
        target = destination / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    return device
