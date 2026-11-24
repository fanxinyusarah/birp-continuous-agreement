import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location('soft', Path(__file__).with_name('soft_floor_wrapper.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
expected_labels = torch.tensor([[[0, 1, 1]]])
module.original = lambda *a, **kw: (expected_labels, torch.tensor([[[0., .6, 1.]]]), {'pseudo_cov': .67, 'pseudo_fg': 1.})
labels, weights, diagnostic = module.soft_floor(None, None, None)
assert torch.equal(labels, expected_labels)
assert torch.allclose(weights, torch.tensor([[[.5, .8, 1.]]]))
assert diagnostic['pseudo_cov'] == 1.0
assert abs(diagnostic['pseudo_fg'] - 2 / 3) < 1e-6
print('Soft-floor label and weighting checks passed')
