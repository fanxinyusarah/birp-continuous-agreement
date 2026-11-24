"""Read-only trajectory/config comparisons. No performance-based selection."""
import argparse
import itertools
import json
from pathlib import Path

from repro_utils import atomic_json, file_hash


def main(root, output):
    root = Path(root)
    runs = []
    for gamma, seed in itertools.product((.5, 1.5, 3.), (1337, 2027, 2028)):
        folder = root / "gamma" / f"gamma{gamma:g}" / f"seed{seed}"
        args = json.loads((folder / "args.json").read_text())
        metrics = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        runs.append(dict(gamma=gamma, seed=seed, args=args,
            environment=json.loads((folder / "run_environment.json").read_text()),
            warmup=[r for r in metrics if r["stage"] == "source_warmup"],
            metrics_sha256=file_hash(folder / "metrics.jsonl")))
    pairs = []
    for a, b in itertools.combinations(runs, 2):
        if a["seed"] != b["seed"]:
            continue
        differences = {k: [a["args"].get(k), b["args"].get(k)]
                       for k in a["args"].keys() | b["args"].keys()
                       if a["args"].get(k) != b["args"].get(k)}
        mismatches = [{"epoch": x["epoch"], "fields": {
            k: [x.get(k), y.get(k)] for k in x.keys() | y.keys() if x.get(k) != y.get(k)}}
            for x, y in zip(a["warmup"], b["warmup"]) if x != y]
        pairs.append(dict(seed=a["seed"], gammas=[a["gamma"], b["gamma"]],
            args_difference=differences,
            hardware=[a["environment"]["gpu"], b["environment"]["gpu"]],
            warmup_exact_match=not mismatches, first_mismatch=mismatches[0] if mismatches else None))
    historical = Path("/scratch/xf810/SegMamba/logs/patient_v1/bidir_repro_audit")
    legacy = []
    for r in runs:
        folder = historical / f"seed{r['seed']}"
        args = json.loads((folder / "args.json").read_text())
        metrics = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        warmup = [x for x in metrics if x["stage"] == "source_warmup"]
        differences = {k: [args.get(k), r["args"].get(k)]
                       for k in args.keys() | r["args"].keys() if args.get(k) != r["args"].get(k)}
        legacy.append(dict(seed=r["seed"], gamma=r["gamma"], args_difference=differences,
                           warmup_exact_match=warmup == r["warmup"],
                           first_epoch=[warmup[0], r["warmup"][0]]))
    atomic_json(output, dict(runs=runs, within_new_gamma_pairs=pairs,
        historical_pairs=legacy, original_source_hash_not_recorded_in_old_checkpoint=True,
        caution="Associations among hardware/configs and trajectories do not establish a unique historical cause."))
    print(json.dumps(dict(pairs=pairs, historical_matching_warmups=sum(x["warmup_exact_match"] for x in legacy)), indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    main(a.root, a.out)
