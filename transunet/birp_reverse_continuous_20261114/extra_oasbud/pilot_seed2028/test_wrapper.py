import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location('external_soft_agreement', Path(__file__).with_name('wrapper.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
class EMA:
    def __call__(self, x):
        return torch.tensor([[[[4., -4.]], [[-4., 4.]]]])
class Prior:
    def __call__(self, x):
        return torch.tensor([[[[.5, 1.]]]])
p, w, d = module.soft_agreement(EMA(), Prior(), None)
assert torch.equal(p, torch.tensor([[[0, 1]]]))
assert torch.isclose(w[0, 0, 0], torch.tensor(.5)) and .99 < w[0, 0, 1] <= 1.
assert not w.requires_grad and d['pseudo_cov'] == 1.
print('external soft-agreement unit test passed')
