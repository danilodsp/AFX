"""Compare afx_audio.egemaps with openSMILE's eGeMAPSv02 functionals.

    pip install opensmile          # research-only licence; not an AFX dependency
    python benchmarks/validate_egemaps.py [--n 20] [--real] [--write-docs]

The synthetic corpus is tests/voice_synth.corpus (glottal pulses through
formant filters with known F0 contour, jitter, shimmer and SNR). With --real
the librosa example LibriSpeech clips are cut into 3 s segments and added as
a second corpus (needs librosa and a network connection once). For every
functional the script reports the Pearson correlation across clips and the
median relative error |afx - opensmile| / |opensmile|, and for the
synthetic clips the error of both implementations against the formant
frequencies, jitter and shimmer the clips were made with. --write-docs
replaces the tables in docs/egemaps.md.
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import afx_audio  # noqa: E402
from tests.voice_synth import corpus  # noqa: E402

SR = 16000
DOC = os.path.join(ROOT, 'docs', 'egemaps.md')


def real_speech(seconds=3.0):
    import librosa

    clips = []
    for key in ('libri1', 'libri2', 'libri3'):
        y, _ = librosa.load(librosa.ex(key), sr=SR)
        step = int(seconds * SR)
        clips += [(f'{key}_{i}', y[i:i + step].astype(np.float64), {}) for i in range(0, len(y) - step + 1, step)]
    return clips


def compare(clips):
    """AFX and openSMILE functionals, (n_clips, 88) each, and AFX time per second of audio."""
    import opensmile

    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)
    names = list(afx_audio.EGEMAPS_NAMES)
    ours, ref, seconds = [], [], []
    for _, x, _ in clips:
        start = time.perf_counter()
        values = afx_audio.egemaps(x, SR)
        seconds.append((time.perf_counter() - start) / (len(x) / SR))
        ours.append([values[n] for n in names])
        ref.append(smile.process_signal(x, SR)[names].to_numpy()[0])
    return names, np.array(ours), np.array(ref, dtype=np.float64), float(np.median(seconds))


def stats(a, b):
    """Pearson r (NaN if either side is constant) and median relative error."""
    r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
    rel = np.median(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))
    return r, rel


def agreement_table(names, results):
    header = '| Functional | ' + ' | '.join(f'r ({k}) | rel. error ({k})' for k in results) + ' |'
    rows = [header, '|---|' + '---:|---:|' * len(results)]
    for i, name in enumerate(names):
        cells = []
        for ours, ref in results.values():
            r, rel = stats(ours[:, i], ref[:, i])
            cells += ['n/a' if np.isnan(r) else f'{r:.3f}', f'{100 * rel:.1f} %']
        rows.append(f'| `{name}` | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def truth_table(names, ours, ref, clips):
    """Mean F1-F3 frequency of each implementation against the formants the clips were synthesized with."""
    rows = ['| Formant | AFX r | AFX mean abs. error | openSMILE r | openSMILE mean abs. error |',
            '|---|---:|---:|---:|---:|']
    for j in range(3):
        true = np.array([params['formants'][j] for _, _, params in clips])
        i = names.index(f'F{j + 1}frequency_sma3nz_amean')
        cells = []
        for values in (ours[:, i], ref[:, i]):
            cells += [f'{np.corrcoef(values, true)[0, 1]:.3f}', f'{np.mean(np.abs(values - true)):.0f} Hz']
        rows.append(f'| F{j + 1} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def voice_quality_table(names, ours, ref, clips):
    """Jitter and shimmer means of each implementation against the realized values of the synthetic clips."""
    rows = ['| Functional | AFX r | AFX slope | AFX mean abs. error | openSMILE r | openSMILE slope '
            '| openSMILE mean abs. error |', '|---|---:|---:|---:|---:|---:|---:|']
    for key, name, scale, unit in (('jitter', 'jitterLocal_sma3nz_amean', 100, ' %'),
                                   ('shimmer_db', 'shimmerLocaldB_sma3nz_amean', 1, ' dB')):
        true = np.array([params['truth'][key] for _, _, params in clips]) * scale
        i = names.index(name)
        cells = []
        for values in (ours[:, i] * scale, ref[:, i] * scale):
            cells += [f'{np.corrcoef(values, true)[0, 1]:.3f}', f'{np.polyfit(true, values, 1)[0]:.2f}',
                      f'{np.mean(np.abs(values - true)):.2f}{unit}']
        rows.append(f'| `{name}` | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def replace_block(doc, tag, text):
    start, end = f'<!-- {tag} start -->', f'<!-- {tag} end -->'
    head, rest = doc.split(start)
    return f'{head}{start}\n{text}\n{end}{rest.split(end)[1]}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20, help='number of synthetic clips')
    parser.add_argument('--real', action='store_true', help='add segments of the librosa LibriSpeech examples')
    parser.add_argument('--write-docs', action='store_true', help='rewrite the tables in docs/egemaps.md')
    args = parser.parse_args()

    corpora = {'synthetic': corpus(args.n, sr=SR)}
    if args.real:
        corpora['speech'] = real_speech()
    results = {}
    for key, clips in corpora.items():
        names, ours, ref, rtf = compare(clips)
        results[key] = (ours, ref)
        r = np.array([stats(ours[:, i], ref[:, i])[0] for i in range(len(names))])
        print(f'{key}: {len(clips)} clips, {rtf * 1000:.0f} ms per second of audio, '
              f'r > 0.9 for {np.sum(r > 0.9)} / {len(names)} functionals, r > 0.7 for {np.sum(r > 0.7)}')
    agreement = agreement_table(names, results)
    truth = truth_table(names, *results['synthetic'], corpora['synthetic'])
    voice_quality = voice_quality_table(names, *results['synthetic'], corpora['synthetic'])
    print(agreement)
    print(truth)
    print(voice_quality)
    if args.write_docs:
        with open(DOC, encoding='utf-8') as f:
            doc = f.read()
        doc = replace_block(doc, 'agreement table', agreement)
        doc = replace_block(doc, 'formant truth table', truth)
        doc = replace_block(doc, 'jitter shimmer truth table', voice_quality)
        with open(DOC, 'w', encoding='utf-8', newline='\n') as f:
            f.write(doc)


if __name__ == '__main__':
    main()
