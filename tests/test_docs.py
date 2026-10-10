import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_feature_reference_is_current():
    script = os.path.join(ROOT, 'docs', 'gen_features.py')
    result = subprocess.run([sys.executable, script, '--check'], capture_output=True)
    assert result.returncode == 0, 'docs/features.md is stale: run python docs/gen_features.py'


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding='utf-8') as f:
        return f.read()


def test_framing_docs_name_every_feature_without_center():
    import inspect

    import afx_audio

    uncentered = sorted(
        name for name, fn in afx_audio.FEATURE_MAP.items()
        if 'hop_length' in inspect.signature(fn).parameters
        and 'center' not in inspect.signature(fn).parameters
    )
    framing = _read('docs', 'configuration.md').split('## Framing')[1].split('\n## ')[0]
    readme = _read('README.md')
    for name in uncentered:
        assert f'except `{name}`' in framing
        assert f'`{name}` is always centered' in readme
