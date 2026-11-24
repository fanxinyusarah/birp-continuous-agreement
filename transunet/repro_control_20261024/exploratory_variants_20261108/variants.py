"""Two fixed exploratory alternatives, wrapping the frozen BiRP trainer."""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import train_repro

original = train_repro.build_reliable_labels
MODE = os.environ.get('BIRP_VARIANT')
if MODE not in ('foreground_floor', 'soft_agreement'):
    raise ValueError('Unexpected BIRP_VARIANT')


@torch.no_grad()
def labels_and_weights(ema, prior, x, *args, **kwargs):
    if MODE == 'foreground_floor':
        pseudo, selected_weight, diagnostics = original(ema, prior, x, *args, **kwargs)
        base = torch.where(pseudo == 1, 0.75, 0.25)
        weight = base + (1.0 - base) * selected_weight
        diagnostics = dict(diagnostics)
    else:
        p = torch.softmax(ema(x), dim=1)
        q = prior(x)[:, 0].float().clamp(0, 1)
        fg = p[:, 1].float()
        pseudo = (fg >= 0.5).long()
        certainty = (2 * (q - 0.5).abs()).clamp(0, 1)
        agreement = (1 - (fg - q).abs()).clamp(0, 1)
        reliability = certainty * agreement
        weight = 0.5 + 0.5 * reliability
        diagnostics = dict(reliability=float(reliability.mean().item()),
                           agreement=float(agreement.mean().item()),
                           fm_fg=float((q >= 0.85).float().mean().item()), routed_to_fm=0.0)
    diagnostics['pseudo_cov'] = 1.0
    diagnostics['pseudo_fg'] = float(pseudo.float().mean().item())
    return pseudo, weight, diagnostics


if __name__ == '__main__':
    train_repro.build_reliable_labels = labels_and_weights
    train_repro.main()
