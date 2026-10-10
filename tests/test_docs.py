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


def test_baselines_prose_quotes_the_table_std_range():
    import re

    text = _read('docs', 'baselines.md')
    table = text.split('<!-- results start -->')[1].split('<!-- results end -->')[0]
    stds = [float(s) for s in re.findall(r'± (\d\.\d+)', table)]
    prose = ' '.join(text.split('## Reading the numbers')[1].split())
    assert f'fold standard deviation is {min(stds):.3f} to {max(stds):.3f}' in prose


def test_librosa_page_lists_every_parity_framing():
    import ast

    tree = ast.parse(_read('tests', 'test_librosa_parity.py'))
    pairs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and all(isinstance(e, ast.Constant) for e in node.elts):
            sizes = [e.value for e in node.elts if type(e.value) is int and 100 <= e.value < 8000]
            if len(sizes) == 2:
                pairs.add(tuple(sizes))
    text = ' '.join(_read('docs', 'librosa.md').split())
    assert len(pairs) >= 4
    assert [p for p in sorted(pairs) if f'{p[0]} / {p[1]}' not in text] == []
