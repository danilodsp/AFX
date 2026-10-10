import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_feature_reference_is_current():
    script = os.path.join(ROOT, 'docs', 'gen_features.py')
    result = subprocess.run([sys.executable, script, '--check'], capture_output=True)
    assert result.returncode == 0, 'docs/features.md is stale: run python docs/gen_features.py'
