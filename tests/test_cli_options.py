"""
afx command-line usage errors and --describe output.
"""
import pytest

from afx_audio import cli


def test_describe_lists_when_a_feature_is_nan(capsys):
    assert cli.main(['--describe', 'sample_entropy']) == 0
    assert 'NaN for: ' in capsys.readouterr().out


def test_input_folder_is_required(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(['-q'])
    assert exc.value.code == 2
    assert 'input_folder is required' in capsys.readouterr().err


def test_empty_feature_list_is_a_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([str(tmp_path), '--features', ' , ', '-q'])
    assert exc.value.code == 2
    assert '--features needs at least one feature name' in capsys.readouterr().err
