# Contributing to AFX

Bug reports, fixes and new features are welcome. For anything larger than a
small fix, open an issue first so we can agree on the approach.

## Development install

```bash
git clone https://github.com/danilodsp/AFX.git
cd AFX
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

The `dev` extra pulls in the test, torch, notebook and docs dependencies plus
ruff, mypy and build. Python 3.10 or newer is required.

## Checks

These are the same checks CI runs on every pull request:

```bash
ruff check .
mypy afx_audio
python -m pytest
python docs/gen_features.py --check
mkdocs build --strict
```

The torch tests are skipped when PyTorch is not installed and the librosa
parity tests when librosa is not installed. To run the notebooks as tests, set
`AFX_RUN_NOTEBOOKS=1`. `pre-commit run --all-files` runs ruff, mypy and the
file checks and strips notebook outputs.

## Adding a feature

1. Write the extractor in the matching module under `afx_audio/extractors/`.
   It takes `(signal, sr)` followed by keyword-only parameters and returns
   `{name: ndarray}` with frames on the last axis, plus frame metadata when
   called with `return_metadata=True`. Reject bad input with a
   `ValueError` or `TypeError` that says how to fix it.
2. Add it to `FEATURE_MAP` in `afx_audio/extract_all.py` and give it a group,
   units and row count in `_INFO` in `afx_audio/registry.py`.
3. Add tests under `tests/` for the output shape and a signal with a known
   answer (a tone, white noise, an impulse). `tests/test_edge_cases.py` runs
   every `FEATURE_MAP` entry on silence, DC and very short input: the output
   must be non-empty and free of inf, raise no warnings (NaN marks an undefined
   frame), and match `metadata['times']` in length. If librosa has an equivalent, add a parity test to
   `tests/test_librosa_parity.py` and list the feature in `docs/librosa.md`.
   If the feature has a torch version in `afx_audio/torch/`, test that it
   matches the NumPy one.
4. Regenerate the feature reference with `python docs/gen_features.py` and
   commit `docs/features.md`.
5. Add a line to `CHANGELOG.md` under the unreleased version.

Cite the paper or standard the definition comes from in the docstring when the
formula is not obvious.

## Pull requests

Keep each pull request to one change and include tests for it. Commit messages
use the imperative mood ("Add spectral spread", not "Added ..."), with a short
first line and a body explaining why when it is not obvious. Changes that alter
the numbers a feature returns go under "Breaking changes" in the changelog.

## Reporting bugs

Open an issue with the AFX, Python, NumPy and SciPy versions, the smallest
code that reproduces the problem and the full traceback. For security issues,
see [SECURITY.md](SECURITY.md).

By contributing, you agree that your contributions are licensed under the MIT
License and that you follow the [Code of Conduct](CODE_OF_CONDUCT.md).
