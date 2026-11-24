"""Exploratory BiRP soft-floor control; do not modify the frozen trainer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import train_repro

original = train_repro.build_reliable_labels
FLOOR = 0.5


def soft_floor(ema_model, autosamus, x_target, *args, **kwargs):
    pseudo, weight, diagnostics = original(ema_model, autosamus, x_target, *args, **kwargs)
    weight = FLOOR + (1.0 - FLOOR) * weight
    diagnostics = dict(diagnostics)
    diagnostics['pseudo_cov'] = 1.0
    diagnostics['pseudo_fg'] = float(pseudo.float().mean().item())
    return pseudo, weight, diagnostics


if __name__ == '__main__':
    train_repro.build_reliable_labels = soft_floor
    train_repro.main()
