import importlib.metadata
import os
import re
import subprocess
import sys

import pytest

import afx_audio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_BLOCK_SKLEARN = "import sys; sys.modules['sklearn'] = None\n"


def _run(code):
    # sys.path[0] is '' under -c, so a source checkout wins over a stale install
    return subprocess.run([sys.executable, '-c', _BLOCK_SKLEARN + code], capture_output=True, text=True, cwd=ROOT)


def test_star_import_without_sklearn():
    result = _run('from afx_audio import *\nprint(extract_all_features.__name__)')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'extract_all_features'


def test_feature_info_does_not_import_torch():
    result = _run("import sys, afx_audio\nafx_audio.feature_info('mfcc')\nprint('torch' in sys.modules)")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'False'


def test_lazy_name_without_sklearn_raises_install_hint():
    result = _run('import afx_audio\nafx_audio.AFXFeatureExtractor')
    assert result.returncode != 0
    assert 'afx-audio[sklearn]' in result.stderr


def test_all_names_resolve():
    for name in afx_audio.__all__:
        assert hasattr(afx_audio, name), name


def test_dir_lists_lazy_names():
    names = dir(afx_audio)
    assert 'AFXFeatureExtractor' in names
    assert set(afx_audio.__all__) <= set(names)


def test_lazy_name_resolves_with_sklearn():
    pytest.importorskip('sklearn')
    from afx_audio.sklearn import AFXFeatureExtractor
    assert afx_audio.AFXFeatureExtractor is AFXFeatureExtractor


def test_unknown_attribute():
    with pytest.raises(AttributeError, match='no_such_name'):
        afx_audio.no_such_name  # noqa: B018


def _read(name):
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        pytest.skip(f'{name} not available (installed tests without the source tree)')
    with open(path, encoding='utf-8') as f:
        return f.read()


def _installed_version():
    try:
        return importlib.metadata.version('afx-audio')
    except importlib.metadata.PackageNotFoundError:
        pytest.skip('afx-audio is not installed, __version__ is a placeholder')


def test_version_matches_installed_metadata():
    assert afx_audio.__version__ == _installed_version()


def test_version_matches_pyproject():
    # Regex rather than tomllib, which needs Python 3.11
    match = re.search(r'^\[project\][^\[]*?^version\s*=\s*"([^"]+)"', _read('pyproject.toml'), re.M | re.S)
    assert match, 'no [project] version in pyproject.toml'
    assert match.group(1) == _installed_version(), 'reinstall the package after a version bump'


def test_version_matches_citation():
    match = re.search(r'^version:\s*"?([^"\s]+)"?\s*$', _read('CITATION.cff'), re.M)
    assert match, 'no version in CITATION.cff'
    assert match.group(1) == _installed_version()


def test_version_without_installed_metadata():
    result = _run(
        'import importlib.metadata as m\n'
        'def missing(name):\n'
        '    raise m.PackageNotFoundError(name)\n'
        'm.version = missing\n'
        'import afx_audio\n'
        'print(afx_audio.__version__)'
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '0.0.0'


def test_unknown_attribute_error_names_the_module():
    with pytest.raises(AttributeError, match="module 'afx_audio' has no attribute 'nope'"):
        afx_audio.nope  # noqa: B018


def test_feature_counts_match_the_registry():
    description = re.search(r'^description\s*=\s*"([^"]+)"', _read('pyproject.toml'), re.M).group(1)
    readme = _read('README.md').split('\n## ')[0]
    for text in (description, readme):
        counts = re.findall(r'(?<![\w.])(\d+) (?=[^.]*?features)', text)
        assert counts and all(int(n) == len(afx_audio.FEATURE_MAP) for n in counts), text
    assert 'eGeMAPS' in description
