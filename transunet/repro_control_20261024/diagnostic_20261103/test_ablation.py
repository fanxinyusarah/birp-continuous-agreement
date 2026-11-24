import importlib.util
import os
from pathlib import Path

import torch

path = Path(__file__).with_name('ablation_wrapper.py')
os.environ['BIRP_ABLATION'] = 'prior_same_gate'
spec = importlib.util.spec_from_file_location('ablation', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class Prior:
    def __call__(self, x):
        return torch.tensor([[[[0.95, 0.05], [0.8, 0.2]]]])

class Teacher:
    def __call__(self, x):
        return torch.tensor([[[[-5., 5.], [5., 5.]], [[5., -5.], [-5., -5.]]]])

x = torch.zeros((1, 1, 2, 2))
args = (Teacher(), Prior(), x, 'bidir', 0.9, 0.85, 0.12, 0.6, 0.55, 1.0, 0.0, 1.5, 0.05)
ema, mask, _ = module.original(*args)
prior, prior_mask, diag = module.controlled_labels(*args)
assert torch.equal(mask, prior_mask)
assert mask.sum() > 0
assert torch.equal(prior, (Prior()(x)[:, 0] >= 0.5).long())
assert not torch.equal(ema, prior)
assert diag['pseudo_cov'] == (mask > 0).float().mean().item()
module.mode = 'ema_all'
ema2, full, diag = module.controlled_labels(*args)
assert torch.equal(ema, ema2)
assert torch.equal(full, torch.ones_like(mask))
assert diag['pseudo_cov'] == 1.0
print('label/mask control checks passed')
