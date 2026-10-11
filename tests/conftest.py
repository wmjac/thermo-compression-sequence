import importlib.util
from pathlib import Path
import pytest
import torch

torch.set_num_threads(1)


@pytest.fixture
def reference_module():
    def load(filename):
        path = Path(__file__).resolve().parents[1] / "reference_sources" / filename
        spec = importlib.util.spec_from_file_location("reference_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return load
