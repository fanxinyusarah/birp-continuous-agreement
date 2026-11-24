import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location('external_soft_floor', Path(__file__).with_name('wrapper.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
labels = torch.tensor([[[0, 1, 1]]])
module.original = lambda *a, **k: (labels, torch.tensor([[[0., .6, 1.]]]), {})
p, w, d = module.soft_floor(None, None, None)
assert torch.equal(p, labels) and torch.allclose(w, torch.tensor([[[.5, .8, 1.]]]))
assert d['pseudo_cov'] == 1. and abs(d['pseudo_fg'] - 2/3) < 1e-6
print('external soft-floor unit test passed')
