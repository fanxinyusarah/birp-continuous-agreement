"""Instrument the unchanged legacy path without changing iterator/RNG timing."""
import argparse
import json
import sys
from pathlib import Path

import torch

from repro_utils import atomic_json, digest, rng_state, runtime_environment


def main(args):
    root = Path(args.root)
    sys.path.insert(0, str(root / "legacy"))
    import train_birp_hard_prior as legacy
    template = json.loads(Path(args.template).read_text())
    folder = Path(args.out)
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    trace = {"init": {}, "batches": [], "forwards": [], "steps": []}
    student_ref = []
    original_build, original_prior = legacy.build_transunet, legacy.FrozenAutoSAMUS

    def build(*a, **kw):
        student = original_build(*a, **kw)
        trace["init"]["student"] = digest(student.state_dict())
        trace["init"]["after_student_rng"] = digest(rng_state())
        student_ref.append(student)
        def hook(_model, inputs, logits):
            if len(trace["forwards"]) < args.steps:
                trace["forwards"].append(dict(input=digest(inputs[0]), logits=digest(logits),
                    logits_sample=logits.detach().float().flatten()[:16].cpu().tolist()))
        student.register_forward_hook(hook)
        return student

    def prior(*a, **kw):
        result = original_prior(*a, **kw)
        trace["init"]["after_prior_rng"] = digest(rng_state())
        trace["environment"] = runtime_environment()
        return result

    class Iterator:
        def __init__(self, actual, stream):
            self.actual, self.stream, self.count = actual, stream, 0
        def __iter__(self):
            return self
        def __next__(self):
            values = next(self.actual)
            if self.count < args.steps:
                trace["batches"].append(dict(stream=self.stream, index=self.count,
                    image=digest(values[0]), labels=digest(values[1]) if len(values) == 3 else None,
                    ids=list(values[-1]), cpu_rng=digest(torch.get_rng_state())))
            self.count += 1
            return values

    class Loader(legacy.DataLoader):
        def __iter__(self):
            # Construct the original iterator immediately; a yielding wrapper
            # would incorrectly postpone target base-seed consumption.
            actual = super().__iter__()
            dataset = self.dataset
            stream = "target" if hasattr(dataset, "keys") else dataset.split
            return Iterator(actual, stream)

    class Scaler(legacy.GradScaler):
        def step(self, optimizer, *a, **kw):
            result = super().step(optimizer, *a, **kw)
            trace["steps"].append(dict(model=digest(student_ref[0].state_dict()),
                                        scaler=self.state_dict()))
            return result

    legacy.build_transunet, legacy.FrozenAutoSAMUS = build, prior
    legacy.DataLoader, legacy.GradScaler = Loader, Scaler
    original_save = torch.save
    def trace_save(value, path, *a, **kw):
        if isinstance(value, dict) and "model" in value and "optimizer" in value:
            atomic_json(str(path) + ".logical_hashes.json", {
                key: digest(value[key]) for key in ("model", "ema_model", "optimizer")})
            return
        return original_save(value, path, *a, **kw)
    # Short diagnostics need logical fingerprints, not multi-GB checkpoint copies.
    legacy.torch.save = trace_save
    template.update(num_epochs=1, max_steps=args.steps, agree_power=args.gamma,
                    seed=1337, skip_target_eval=True, output_dir=str(folder / "training"))
    argv = ["train_birp_hard_prior.py"]
    for key, value in template.items():
        if isinstance(value, bool):
            if value:
                argv.append("--" + key)
        else:
            argv.extend(["--" + key, str(value)])
    sys.argv = argv
    legacy.main()
    trace["metrics"] = [json.loads(s) for s in (folder / "training/metrics.jsonl").read_text().splitlines()]
    atomic_json(folder / "trace.json", trace)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--steps", type=int, default=4)
    main(p.parse_args())
