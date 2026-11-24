import json
import statistics as stats
from pathlib import Path

ROOT = Path(__file__).parent / 'runs'
SEEDS = (1337, 2027, 2028)
METHODS = ('gamma1.5', 'hard_prior', 'gamma3', 'binary_reliability')
records, summaries, logs = {}, {}, {}
test_hashes = None
for seed in SEEDS:
    for method in METHODS:
        root = ROOT / f'seed{seed}' / method
        summary = json.loads((root / 'locked_evaluation/summary.json').read_text())
        rows = json.loads((root / 'locked_evaluation/per_case.json').read_text())
        assert len(rows) == summary['n_cases'] == 376
        data = {x['case']: x for x in rows}
        assert len(data) == 376
        if test_hashes is None:
            test_hashes = summary['test_files_sha256']
        assert summary['test_files_sha256'] == test_hashes
        assert set(data) == set(test_hashes)
        summaries[seed, method], records[seed, method] = summary, data
        logs[seed, method] = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]

cases = sorted(test_hashes)
for case in cases:
    fractions = [records[s, m][case]['target_fraction'] for s in SEEDS for m in METHODS]
    assert max(fractions) - min(fractions) < 1e-12

print('Identical image/label hashes across all runs:', len(cases))
for method in METHODS:
    print('\n', method)
    for seed in SEEDS:
        summary = summaries[seed, method]
        selected = summary['selected_epoch']
        train = [x for x in logs[seed, method] if x['epoch'] <= selected and x['stage'] != 'source_warmup']
        print('seed', seed, 'epoch', selected, 'dice', round(summary['mean']['dice'], 5),
              'iou', round(summary['mean']['iou'], 5), 'hd95_px', round(summary['mean']['hd95_px'], 3),
              'logged_mean_pseudo_cov', round(stats.mean(x['pseudo_cov'] for x in train), 4) if train else None,
              'empty_predictions', sum(records[seed, method][c]['empty_prediction'] for c in cases))

def avg(method, case, metric):
    return stats.mean(records[s, method][case][metric] for s in SEEDS)

for method in ('gamma1.5', 'gamma3', 'binary_reliability'):
    diff = {c: avg(method, c, 'dice') - avg('hard_prior', c, 'dice') for c in cases}
    print('\n', method, 'minus hard_prior case mean dice:', round(stats.mean(diff.values()), 5),
          'wins/losses/ties:', sum(x > 1e-12 for x in diff.values()),
          sum(x < -1e-12 for x in diff.values()), sum(abs(x) <= 1e-12 for x in diff.values()))
    ordered = sorted(cases, key=lambda c: records[SEEDS[0], method][c]['target_fraction'])
    for i in range(4):
        group = ordered[i * 94:(i + 1) * 94]
        print('lesion area quartile', i + 1, 'n', len(group),
              'area_range', tuple(round(records[SEEDS[0], method][c]['target_fraction'], 4) for c in (group[0], group[-1])),
              'mean_dice_difference', round(stats.mean(diff[c] for c in group), 5))
