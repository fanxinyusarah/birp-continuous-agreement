"""Exploratory aggregate-only audit; execute beside the remote experiment."""
import json
import statistics as st
from pathlib import Path

ROOT = Path('/scratch/xf810/birp_repro_control_20261024')
SEEDS = (1337, 2027, 2028)
METHODS = ('gamma1.5', 'hard_prior', 'ema_all')
results, hashes = {}, None
for method in METHODS:
    for seed in SEEDS:
        base = (ROOT / 'diagnostic_20261104/runs' / f'seed{seed}' / method
                if method == 'ema_all' else ROOT / 'runs' / f'seed{seed}' / method)
        summary = json.loads((base / 'locked_evaluation/summary.json').read_text())
        cases = json.loads((base / 'locked_evaluation/per_case.json').read_text())
        data = {case['case']: case for case in cases}
        assert len(cases) == len(data) == summary['n_cases'] == 376
        hashes = summary['test_files_sha256'] if hashes is None else hashes
        assert hashes == summary['test_files_sha256'] and set(data) == set(hashes)
        logs = [json.loads(x) for x in (base / 'metrics.jsonl').read_text().splitlines()]
        epochs = [x for x in logs if x['epoch'] <= summary['selected_epoch'] and x['stage'] != 'source_warmup']
        results[method, seed] = (data, summary, epochs)
        print(method, seed, 'selected_epoch', summary['selected_epoch'],
              'dice', round(summary['mean']['dice'], 5),
              'training_selected_fraction', round(st.mean(x['pseudo_cov'] for x in epochs), 5),
              'training_selected_foreground_fraction', round(st.mean(x['pseudo_fg'] for x in epochs), 5))

cases = sorted(hashes)
for name in cases:
    fractions = [results[m, s][0][name]['target_fraction'] for m in METHODS for s in SEEDS]
    assert max(fractions) - min(fractions) < 1e-12
ordered = sorted(cases, key=lambda c: results['gamma1.5', SEEDS[0]][0][c]['target_fraction'])
for comparator in ('gamma1.5', 'hard_prior'):
    print('\nEMA-all minus', comparator)
    for quartile in range(4):
        group = ordered[quartile * 94:(quartile + 1) * 94]
        values = [st.mean(results['ema_all', s][0][c]['dice'] - results[comparator, s][0][c]['dice'] for s in SEEDS) for c in group]
        areas = [results['gamma1.5', SEEDS[0]][0][c]['target_fraction'] for c in group]
        print('quartile', quartile + 1, 'n', len(group), 'area_range', round(min(areas), 5), round(max(areas), 5),
              'mean_difference', round(st.mean(values), 5), 'wins', sum(x > 1e-12 for x in values),
              'losses', sum(x < -1e-12 for x in values))
print('\nDescriptive image-level summaries only; no patient-independent inference or target-label-based tuning.')
