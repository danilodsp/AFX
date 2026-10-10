"""ESC-50 classification baselines for the bundled presets.

    python benchmarks/baselines.py ESC50_ROOT [--cache DIR] [--jobs N] [--write-docs]

ESC50_ROOT is either the GitHub release (audio/*.wav and meta/esc50.csv) or
the Harvard Dataverse release (doi:10.7910/DVN/YDEPUT, one folder per class
such as "101 - Dog/1-100032-A.ogg"). In both the fold is the first field of
the file name.

For every bundled preset id and the bundled config.json, features are
extracted with extract_features_from_folder at the config's own sample rate
(parquet, dataset_stats on) into --cache, one file per config named after
its config hash. A config whose parquet and timing record exist is not
extracted again, so an interrupted run resumes where it stopped. Each
feature table is then scored with the official 5-fold split by two fixed
pipelines: median imputation, standardization and a logistic regression
(C=1, max_iter=5000), and the same front with a 500-tree random forest.
--write-docs replaces the generated blocks of docs/baselines.md.
"""
import argparse
import copy
import csv
import json
import os
import platform
import sys
import tempfile
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, 'docs', 'baselines.md')
CONFIG_IDS = ('speech@1', 'bioacoustics@1', 'music@1', 'machinery@1', 'egemaps@1', 'config.json')
AUDIO_EXTS = ('.wav', '.ogg', '.flac')


def load_named_config(name):
    """A bundled preset id, or 'config.json' for the bundled default config."""
    import afx_audio as afx

    if name == 'config.json':
        return afx.load_config(os.path.join(os.path.dirname(afx.__file__), 'config.json'))
    return afx.load_preset(name)


def esc50_index(root):
    """
    (audio folder, {path relative to it: (fold, label)}) for either ESC-50 layout.

    Keys use forward slashes, as extract_features_from_folder writes them.
    """
    meta = os.path.join(root, 'meta', 'esc50.csv')
    if os.path.isfile(meta):
        with open(meta, newline='', encoding='utf-8') as f:
            index = {row['filename']: (int(row['fold']), row['category']) for row in csv.DictReader(f)}
        return os.path.join(root, 'audio'), index
    index = {}
    for label in sorted(os.listdir(root)):
        folder = os.path.join(root, label)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(AUDIO_EXTS):
                index[f'{label}/{name}'] = (int(name.split('-')[0]), label)
    return root, index


def extract(name, config, audio_dir, cache_dir, n_jobs):
    """
    Extract one config into cache_dir unless already done; returns the parquet path and timing record.

    The timing record is written last, so a parquet without one is from an
    interrupted run and is extracted again.
    """
    import afx_audio as afx

    chash = afx.config_hash(afx.validate_config(copy.deepcopy(config)))
    stem = name.replace('@', '-').replace('.json', '')
    out = os.path.join(cache_dir, f'{stem}_{chash}.parquet')
    timing_path = f'{out}.timing.json'
    if os.path.isfile(out) and os.path.isfile(timing_path):
        with open(timing_path, encoding='utf-8') as f:
            return out, json.load(f)
    start = time.perf_counter()
    afx.extract_features_from_folder(audio_dir, config, out, n_jobs=n_jobs, verbose=False, dataset_stats=True)
    timing = {'wall_s': time.perf_counter() - start, 'n_jobs': n_jobs}
    with open(timing_path, 'w', encoding='utf-8') as f:
        json.dump(timing, f)
    return out, timing


def make_models(n_trees=500, n_jobs=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def front():
        return [SimpleImputer(strategy='median'), StandardScaler()]

    return {
        'logreg': make_pipeline(*front(), LogisticRegression(max_iter=5000, C=1.0)),
        'rf': make_pipeline(*front(), RandomForestClassifier(n_estimators=n_trees, random_state=0, n_jobs=n_jobs)),
    }


def evaluate(parquet, index, n_trees=500, n_jobs=None):
    """Accuracy per official fold for each model, plus the column and NaN counts of the feature table."""
    import pandas as pd
    from sklearn.model_selection import PredefinedSplit, cross_val_score

    import afx_audio as afx

    df = pd.read_parquet(parquet)
    missing = sorted(set(index) - set(df.index))
    df = df.loc[[k for k in index if k in df.index]]
    X = df.to_numpy(dtype=np.float64)
    folds = np.array([index[k][0] for k in df.index])
    y = np.array([index[k][1] for k in df.index])
    stats = afx.load_stats(parquet)
    nan_count = stats['nan_count']
    provenance = afx.read_provenance(parquet)
    result = {
        'n_clips': len(y),
        'n_missing': len(missing),
        'n_columns': X.shape[1],
        'nan_columns': int(np.sum(nan_count > 0)),
        'all_nan_columns': int(np.sum(stats['count'] == 0)),
        'nan_values': float(np.mean(np.isnan(X))),
        'config_hash': provenance['config_hash'],
        'preset_id': provenance['preset_id'],
        'sample_rate': provenance['sample_rate'],
        'resampling': provenance['resampling'],
        'versions': provenance['versions'],
        'scores': {},
    }
    # Fold labels start at 1; PredefinedSplit wants 0-based test fold ids
    split = PredefinedSplit(np.unique(folds, return_inverse=True)[1])
    for key, model in make_models(n_trees, n_jobs).items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            scores = cross_val_score(model, X, y, cv=split)
        result['scores'][key] = {
            'folds': scores.tolist(),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'warnings': sorted({type(w.message).__name__ for w in caught}),
        }
    return result


def results_table(results):
    rows = [
        '| Config | Config hash | Sample rate | Columns | NaN columns | Extraction (s) | '
        'Logistic regression | Random forest |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name, r in results.items():
        lr, rf = r['scores']['logreg'], r['scores']['rf']
        rows.append(
            f"| `{name}` | `{r['config_hash']}` | {r['sample_rate']} | {r['n_columns']} | "
            f"{r['nan_columns']} ({r['nan_columns'] / r['n_columns']:.0%}) | {r['wall_s']:.0f} | "
            f"{lr['mean']:.3f} ± {lr['std']:.3f} | {rf['mean']:.3f} ± {rf['std']:.3f} |"
        )
    return '\n'.join(rows)


def fold_table(results):
    rows = ['| Config | Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |', '|---|---|---:|---:|---:|---:|---:|']
    for name, r in results.items():
        for key, label in (('logreg', 'Logistic regression'), ('rf', 'Random forest')):
            rows.append(f'| `{name}` | {label} | ' + ' | '.join(f'{s:.3f}' for s in r['scores'][key]['folds']) + ' |')
    return '\n'.join(rows)


def _cpu_model():
    try:
        with open('/proc/cpuinfo', encoding='utf-8') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def environment_text(results, n_jobs):
    import pandas
    import pyarrow
    import sklearn

    first = next(iter(results.values()))
    v = first['versions']
    rates = ', '.join(f'{k} Hz: {n}' for k, n in first['resampling']['native_sample_rates'].items())
    return (
        f'Machine: {_cpu_model()}, {os.cpu_count()} logical CPUs, {platform.system()} {platform.release()}; '
        f'extraction with `n_jobs={n_jobs}`.\n'
        f"Versions: afx-audio {v['afx_audio']}, Python {v['python']}, NumPy {v['numpy']}, SciPy {v['scipy']}, "
        f"soundfile {v['soundfile']} (libsndfile {v['libsndfile']}), scikit-learn {sklearn.__version__}, "
        f'pandas {pandas.__version__}, pyarrow {pyarrow.__version__}.\n'
        f"Clips: {first['n_clips']} scored, {first['n_missing']} missing; native sample rates ({rates}), "
        'resampled with `scipy.signal.resample_poly` to each config\'s rate.'
    )


def replace_block(doc, tag, text):
    start, end = f'<!-- {tag} start -->', f'<!-- {tag} end -->'
    head, rest = doc.split(start)
    return f'{head}{start}\n{text}\n{end}{rest.split(end)[1]}'


def available_cores():
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def run(root, cache_dir, config_ids=CONFIG_IDS, n_jobs=None, n_trees=500, log=print):
    """Extract (or reuse) and score every config; returns {config id: result}."""
    n_jobs = n_jobs or available_cores()
    audio_dir, index = esc50_index(root)
    os.makedirs(cache_dir, exist_ok=True)
    results = {}
    for name in config_ids:
        parquet, timing = extract(name, load_named_config(name), audio_dir, cache_dir, n_jobs)
        log(f'{name}: features in {timing["wall_s"]:.0f} s ({os.path.basename(parquet)})')
        result = evaluate(parquet, index, n_trees=n_trees, n_jobs=n_jobs)
        result.update(wall_s=timing['wall_s'], n_jobs=timing['n_jobs'])
        lr, rf = result['scores']['logreg'], result['scores']['rf']
        log(f'{name}: {result["n_columns"]} columns, logreg {lr["mean"]:.3f} +/- {lr["std"]:.3f}, '
            f'rf {rf["mean"]:.3f} +/- {rf["std"]:.3f}')
        results[name] = result
    with open(os.path.join(cache_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('root', help='ESC-50 folder (GitHub or Dataverse layout)')
    parser.add_argument('--cache', default=os.path.join(tempfile.gettempdir(), 'afx-baselines'),
                        help='folder for the parquet files and results.json')
    parser.add_argument('--jobs', type=int, default=None, help='worker processes (default: available cores)')
    parser.add_argument('--configs', default=','.join(CONFIG_IDS), help='comma-separated preset ids or config.json')
    parser.add_argument('--write-docs', action='store_true', help='rewrite the generated blocks of docs/baselines.md')
    args = parser.parse_args(argv)

    results = run(args.root, args.cache, args.configs.split(','), n_jobs=args.jobs)
    n_jobs = next(iter(results.values()))['n_jobs']
    table, folds, env = results_table(results), fold_table(results), environment_text(results, n_jobs)
    print(table, folds, env, sep='\n\n')
    if args.write_docs:
        with open(DOC, encoding='utf-8') as f:
            doc = f.read()
        for tag, text in (('results', table), ('folds', folds), ('environment', env)):
            doc = replace_block(doc, tag, text)
        with open(DOC, 'w', encoding='utf-8', newline='\n') as f:
            f.write(doc)
    return results


if __name__ == '__main__':
    main(sys.argv[1:])
