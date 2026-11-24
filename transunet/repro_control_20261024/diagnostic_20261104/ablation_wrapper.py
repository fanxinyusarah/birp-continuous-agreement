"""Additional label/mask controls without editing the frozen trainer.

The caller must record this file's hash and the chosen mode separately: the
original trainer's provenance covers its own files, not this wrapper.
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import train_repro

original = train_repro.build_reliable_labels
mode = os.environ.get('BIRP_ABLATION')
if mode not in {'prior_same_gate', 'ema_all'}:
    raise ValueError('BIRP_ABLATION must be prior_same_gate or ema_all')


def controlled_labels(ema_model, autosamus, x_target, *args, **kwargs):
    pseudo, mask, diagnostics = original(ema_model, autosamus, x_target, *args, **kwargs)
    if mode == 'prior_same_gate':
        pseudo = (autosamus(x_target)[:, 0] >= 0.5).long()
    else:
        mask = torch.ones_like(mask)
    diagnostics = dict(diagnostics)
    diagnostics['pseudo_cov'] = float((mask > 0).float().mean().item())
    diagnostics['pseudo_fg'] = float((pseudo.float() * (mask > 0)).sum().item() / (mask > 0).sum().clamp_min(1).item())
    return pseudo, mask, diagnostics


if __name__ == '__main__':
    train_repro.build_reliable_labels = controlled_labels
    train_repro.main()
