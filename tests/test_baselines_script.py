import importlib.util
import json
import os

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip('sklearn')
pytest.importorskip('pyarrow')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = {
    'sample_rate': 8000,
    'features': {'mfcc': {'n_mfcc': 8, 'frame_size': 256, 'hop_length': 128}, 'pitch': {}},
    'aggregation': 'summary',
}


def _load():
    spec = importlib.util.spec_from_file_location('baselines', os.path.join(ROOT, 'benchmarks', 'baselines.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataverse_layout(root, sr=16000):
    """Two classes, two clips per fold in the Dataverse naming ('101 - Tone/3-12-A.ogg')."""
    rng = np.random.default_rng(0)
    t = np.arange(sr // 4) / sr
    for label, make in [('101 - Tone', lambda: np.sin(2 * np.pi * rng.uniform(300, 500) * t)),
                        ('102 - Noise', lambda: rng.standard_normal(len(t)))]:
        (root / label).mkdir()
        for fold in range(1, 6):
            for take in 'AB':
                path = root / label / f'{fold}-{rng.integers(1000)}-{take}.wav'
                sf.write(path, (0.3 * make()).astype(np.float32), sr)


def test_index_layouts(tmp_path):
    baselines = _load()
    _dataverse_layout(tmp_path)
    audio_dir, index = baselines.esc50_index(str(tmp_path))
    assert audio_dir == str(tmp_path) and len(index) == 20
    assert sorted({fold for fold, _ in index.values()}) == [1, 2, 3, 4, 5]

    github = tmp_path / 'gh'
    (github / 'meta').mkdir(parents=True)
    (github / 'meta' / 'esc50.csv').write_text('filename,fold,target,category\n1-100032-A-0.wav,1,0,dog\n')
    audio_dir, index = baselines.esc50_index(str(github))
    assert audio_dir == str(github / 'audio')
    assert index == {'1-100032-A-0.wav': (1, 'dog')}


def test_extract_is_cached_and_evaluated(tmp_path):
    baselines = _load()
    data, cache = tmp_path / 'data', tmp_path / 'cache'
    data.mkdir()
    cache.mkdir()
    _dataverse_layout(data)
    audio_dir, index = baselines.esc50_index(str(data))

    out, timing = baselines.extract('tiny@1', CONFIG, audio_dir, str(cache), n_jobs=1)
    assert os.path.isfile(out) and os.path.isfile(f'{out}.stats.json')
    mtime = os.path.getmtime(out)
    assert baselines.extract('tiny@1', CONFIG, audio_dir, str(cache), n_jobs=1) == (out, timing)
    assert os.path.getmtime(out) == mtime

    result = baselines.evaluate(out, index, n_trees=10, n_jobs=1)
    assert result['n_clips'] == 20 and result['n_missing'] == 0
    assert result['n_columns'] == 2 * 8 + 2
    # pitch is NaN on unvoiced noise clips but defined on the tones
    assert result['nan_columns'] <= 2 and result['all_nan_columns'] == 0
    assert result['resampling']['n_resampled'] == 20
    for key in ('logreg', 'rf'):
        assert len(result['scores'][key]['folds']) == 5
        assert result['scores'][key]['mean'] > 0.9

    result.update(wall_s=timing['wall_s'], n_jobs=1)
    table = baselines.results_table({'tiny@1': result})
    assert f"`{result['config_hash']}`" in table and '| 18 |' in table
    assert baselines.fold_table({'tiny@1': result}).count('\n') == 3

    doc = 'intro\n<!-- results start -->\nold\n<!-- results end -->\noutro\n'
    new = baselines.replace_block(doc, 'results', table)
    assert new == f'intro\n<!-- results start -->\n{table}\n<!-- results end -->\noutro\n'


def test_run_writes_results(tmp_path, monkeypatch):
    baselines = _load()
    data, cache = tmp_path / 'data', tmp_path / 'cache'
    data.mkdir()
    _dataverse_layout(data)
    monkeypatch.setattr(baselines, 'load_named_config', lambda name: CONFIG)
    results = baselines.run(str(data), str(cache), ['tiny@1'], n_jobs=1, n_trees=10, log=lambda msg: None)
    with open(cache / 'results.json', encoding='utf-8') as f:
        assert json.load(f)['tiny@1']['n_columns'] == results['tiny@1']['n_columns']
    assert 'n_jobs=1' in baselines.environment_text(results, 1)
