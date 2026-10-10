"""
The numbered notebooks: stored without outputs, and runnable end to end.

Execution is opt-in (AFX_RUN_NOTEBOOKS=1) because it starts a Jupyter kernel
per notebook. Dataset environment variables are cleared so every notebook
runs on synthetic audio.
"""
import importlib.util
import os
import time
from pathlib import Path

import pytest

nbformat = pytest.importorskip('nbformat')

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / 'notebooks'

# Wall-clock budget per notebook in seconds, kernel start included: about 10x
# the time on a 2-core Linux machine, for slower CI runners and spawn start-up
BUDGET = {
    '01_quickstart.ipynb': 60,
    '02_frames_and_aggregation.ipynb': 60,
    '03_long_recordings_and_custom_features.ipynb': 90,
    '04_batch_and_parallel.ipynb': 120,
    '05_sklearn_and_selection.ipynb': 180,
    '06_torch.ipynb': 180,
    '07_domain_presets.ipynb': 120,
}
REQUIRES = {
    '02_frames_and_aggregation.ipynb': ('matplotlib',),
    '03_long_recordings_and_custom_features.ipynb': ('matplotlib',),
    '04_batch_and_parallel.ipynb': ('pandas', 'pyarrow', 'sklearn'),
    '05_sklearn_and_selection.ipynb': ('sklearn', 'matplotlib'),
    '06_torch.ipynb': ('torch', 'matplotlib'),
    '07_domain_presets.ipynb': ('sklearn', 'matplotlib'),
}
DATA_ENV_VARS = ('ESC_50_dataset', 'UrbanSound8K_dataset', 'dataset_nonspeech7k', 'DATASETS')


def test_numbered_set_is_complete():
    assert sorted(p.name for p in NOTEBOOKS.glob('[0-9][0-9]_*.ipynb')) == sorted(BUDGET)


@pytest.mark.parametrize('path', sorted(NOTEBOOKS.glob('*.ipynb')), ids=lambda p: p.name)
def test_stored_without_outputs(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        assert 'id' in cell
        if cell.cell_type == 'code':
            assert not cell.outputs and cell.execution_count is None, f'{path.name}: clear outputs before committing'


@pytest.mark.skipif(os.environ.get('AFX_RUN_NOTEBOOKS') != '1', reason='set AFX_RUN_NOTEBOOKS=1 to execute notebooks')
@pytest.mark.slow
@pytest.mark.parametrize('name', sorted(BUDGET))
def test_notebook_runs(name, monkeypatch):
    nbclient = pytest.importorskip('nbclient')
    pytest.importorskip('ipykernel')
    for module in REQUIRES.get(name, ()):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f'{name} needs {module}')
    for var in DATA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv('AFX_SYNTHETIC', '1')
    # The kernel must import this checkout even when another copy is installed
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join(filter(None, [str(ROOT), os.environ.get('PYTHONPATH')])))
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')

    before = sorted(NOTEBOOKS.rglob('*'))
    nb = nbformat.read(NOTEBOOKS / name, as_version=4)
    client = nbclient.NotebookClient(nb, timeout=BUDGET[name], kernel_name='python3',
                                     resources={'metadata': {'path': str(NOTEBOOKS)}})
    start = time.perf_counter()
    client.execute()
    elapsed = time.perf_counter() - start
    after = sorted(NOTEBOOKS.rglob('*'))

    assert elapsed < BUDGET[name], f'{name} took {elapsed:.0f} s, budget {BUDGET[name]} s'
    assert after == before, f'{name} wrote into notebooks/: {sorted(set(after) - set(before))}'
