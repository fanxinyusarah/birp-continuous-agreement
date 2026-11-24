"""Fixed exploratory soft-floor version for an external target cohort."""
import os
import sys
from pathlib import Path

root = Path(os.environ['BIRP_EXTERNAL_ROOT'])
sys.path.insert(0, str(root / 'src'))
import train_repro

original = train_repro.build_reliable_labels


def soft_floor(ema, prior, x, *args, **kwargs):
    pseudo, weight, diagnostics = original(ema, prior, x, *args, **kwargs)
    diagnostics = dict(diagnostics)
    diagnostics['pseudo_cov'] = 1.0
    diagnostics['pseudo_fg'] = float(pseudo.float().mean().item())
    return pseudo, 0.5 + 0.5 * weight, diagnostics


if __name__ == '__main__':
    train_repro.build_reliable_labels = soft_floor
    train_repro.main()
