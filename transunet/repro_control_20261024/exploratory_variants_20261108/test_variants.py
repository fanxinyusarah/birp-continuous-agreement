import importlib.util
import os
from pathlib import Path
import torch

os.environ['BIRP_VARIANT'] = 'foreground_floor'
spec = importlib.util.spec_from_file_location('v', Path(__file__).with_name('variants.py'))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
v.original = lambda *a, **kw: (torch.tensor([[[0, 1, 1]]]), torch.tensor([[[0., 0., 1.]]]), {})
p, w, d = v.labels_and_weights(None, None, None)
assert torch.allclose(w, torch.tensor([[[.25, .75, 1.]]])) and d['pseudo_cov'] == 1.
v.MODE = 'soft_agreement'
class EMA:
    def __call__(self, x):
        return torch.tensor([[[[4., -4.]], [[-4., 4.]]]])
class Prior:
    def __call__(self, x):
        return torch.tensor([[[[.5, 1.]]]])
p, w, d = v.labels_and_weights(EMA(), Prior(), None)
assert torch.equal(p, torch.tensor([[[0, 1]]]))
assert torch.isclose(w[0, 0, 0], torch.tensor(.5)) and .99 < w[0, 0, 1] <= 1.
assert not w.requires_grad and d['pseudo_cov'] == 1.
print('Both exploratory variants passed')
