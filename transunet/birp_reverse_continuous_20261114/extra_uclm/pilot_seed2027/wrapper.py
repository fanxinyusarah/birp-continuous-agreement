"""External reproduction of the fixed exploratory continuous-agreement rule."""
import os
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(os.environ['BIRP_EXTERNAL_ROOT']) / 'src'))
import train_repro


@torch.no_grad()
def soft_agreement(ema, prior, x, *args, **kwargs):
    p = torch.softmax(ema(x), dim=1)
    q = prior(x)[:, 0].float().clamp(0, 1)
    fg = p[:, 1].float()
    pseudo = (fg >= 0.5).long()
    certainty = (2 * (q - 0.5).abs()).clamp(0, 1)
    agreement = (1 - (fg - q).abs()).clamp(0, 1)
    reliability = certainty * agreement
    weight = 0.5 + 0.5 * reliability
    return pseudo, weight, dict(pseudo_cov=1.0, pseudo_fg=float(pseudo.float().mean().item()),
        reliability=float(reliability.mean().item()), agreement=float(agreement.mean().item()),
        fm_fg=float((q >= 0.85).float().mean().item()), routed_to_fm=0.0)


if __name__ == '__main__':
    train_repro.build_reliable_labels = soft_agreement
    train_repro.main()
