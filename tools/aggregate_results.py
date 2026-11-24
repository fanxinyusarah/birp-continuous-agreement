"""Print test-set means and sample SDs across the three reported seeds."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, stdev

REPORTED_SEEDS = {1337, 2027, 2028}


def summarize(records):
    groups = defaultdict(dict)
    for record in records:
        key = tuple(record[field] for field in ("pipeline", "setting", "source", "target", "method"))
        seed = record["seed"]
        if seed in groups[key]:
            raise ValueError(f"Duplicate run for {key} seed {seed}")
        groups[key][seed] = record

    for key, runs in sorted(groups.items()):
        if set(runs) != REPORTED_SEEDS:
            raise ValueError(f"Wrong seed set for {key}: {sorted(runs)}")
        metric_names = set.intersection(*(set(run["metrics"]) for run in runs.values()))
        for metric in sorted(metric_names):
            values = [run["metrics"][metric] for run in runs.values()]
            yield (*key, metric, mean(values), stdev(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/available_run_metrics.json")
    args = parser.parse_args()
    print("pipeline\tsetting\tsource\ttarget\tmethod\tmetric\tmean\tsample_sd")
    for *key, avg, sd in summarize(json.loads(args.metrics.read_text())):
        print("\t".join([*key, f"{avg:.4f}", f"{sd:.4f}"]))


if __name__ == "__main__":
    main()
