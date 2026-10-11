"""Time every feature in a config on synthetic audio.

    python benchmarks/bench_features.py [--config PATH] [--seconds 10] [--repeat 5]
    python benchmarks/bench_features.py --markdown --librosa --torch-batch 16

Reports the median wall time per feature in milliseconds over ``--repeat``
runs, after one untimed warm-up call. The STFT cache is cleared before each
call, so every feature pays for its own transform. ``--librosa`` times the
librosa function with the same framing where one exists, ``--torch-batch``
times the torch path on a batch of clips on the CPU, and ``--markdown``
prints a table with the machine and library versions.
"""
import argparse
import inspect
import os
import platform
import time

import numpy as np

import afx_audio


def _param(name, params, key):
    """Value of ``key`` for feature ``name``: the config's, else the extractor's default."""
    if key in params:
        return params[key]
    return inspect.signature(afx_audio.FEATURE_MAP[name]).parameters[key].default


def _librosa_calls(librosa):
    """Feature name -> (label, call(y, sr, name, params)) for features with a librosa counterpart."""

    def stft_kw(name, p):
        return {'n_fft': _param(name, p, 'frame_size'), 'hop_length': _param(name, p, 'hop_length'),
                'center': _param(name, p, 'center')}

    def mfcc(y, sr, name, p):
        return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=_param(name, p, 'n_mfcc'), n_mels=_param(name, p, 'n_mels'),
                                    **stft_kw(name, p))

    def spectral(func, with_sr=True):
        def call(y, sr, name, p):
            return func(y=y, sr=sr, **stft_kw(name, p)) if with_sr else func(y=y, **stft_kw(name, p))
        return call

    f = librosa.feature
    return {
        'zcr': ('feature.zero_crossing_rate', lambda y, sr, n, p: f.zero_crossing_rate(
            y, frame_length=_param(n, p, 'frame_size'), hop_length=_param(n, p, 'hop_length'),
            center=_param(n, p, 'center'))),
        'rms_energy': ('feature.rms', lambda y, sr, n, p: f.rms(
            y=y, frame_length=_param(n, p, 'frame_size'), hop_length=_param(n, p, 'hop_length'),
            center=_param(n, p, 'center'))),
        'spectral_centroid': ('feature.spectral_centroid', spectral(f.spectral_centroid)),
        'spectral_bandwidth': ('feature.spectral_bandwidth', spectral(f.spectral_bandwidth)),
        'spectral_rolloff': ('feature.spectral_rolloff', spectral(f.spectral_rolloff)),
        'spectral_flatness': ('feature.spectral_flatness', spectral(f.spectral_flatness, with_sr=False)),
        'spectral_contrast': ('feature.spectral_contrast', spectral(f.spectral_contrast)),
        'melspectrogram': ('feature.melspectrogram', lambda y, sr, n, p: f.melspectrogram(
            y=y, sr=sr, n_mels=_param(n, p, 'n_mels'), **stft_kw(n, p))),
        'mfcc': ('feature.mfcc', mfcc),
        'mfcc_delta': ('feature.delta(mfcc)', lambda y, sr, n, p: f.delta(mfcc(y, sr, n, p))),
        'mfcc_delta_delta': ('feature.delta(mfcc, order=2)', lambda y, sr, n, p: f.delta(mfcc(y, sr, n, p), order=2)),
        'chroma_stft': ('feature.chroma_stft(tuning=0)', lambda y, sr, n, p: f.chroma_stft(
            y=y, sr=sr, tuning=0.0, **stft_kw(n, p))),
        'tonnetz': ('feature.tonnetz(chroma=chroma_stft)', lambda y, sr, n, p: f.tonnetz(
            chroma=f.chroma_stft(y=y, sr=sr, tuning=0.0, **stft_kw(n, p)))),
        'onset_strength': ('onset.onset_strength', lambda y, sr, n, p: librosa.onset.onset_strength(
            y=y, sr=sr, n_fft=_param(n, p, 'frame_size'), hop_length=_param(n, p, 'hop_length'))),
        'pitch': ('yin *', lambda y, sr, n, p: librosa.yin(
            y, fmin=_param(n, p, 'fmin'), fmax=_param(n, p, 'fmax'), sr=sr,
            frame_length=_param(n, p, 'frame_size'), hop_length=_param(n, p, 'hop_length'),
            center=_param(n, p, 'center'), pad_mode='constant')),
        # True constant-Q transforms, not the STFT approximation AFX computes
        'cqt': ('cqt *', lambda y, sr, n, p: np.abs(librosa.cqt(
            y, sr=sr, hop_length=_param(n, p, 'hop_length'), n_bins=_param(n, p, 'n_bins'),
            bins_per_octave=_param(n, p, 'bins_per_octave')))),
        'chroma_cqt': ('feature.chroma_cqt *', lambda y, sr, n, p: f.chroma_cqt(
            y=y, sr=sr, hop_length=_param(n, p, 'hop_length'))),
    }


def _time(func, repeat, before=None):
    """Median and interquartile range (ms) of ``repeat`` timed calls after one warm-up."""
    if before:
        before()
    func()
    times = []
    for _ in range(repeat):
        if before:
            before()
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    q25, q50, q75 = 1000 * np.percentile(times, [25, 50, 75])
    return float(q50), float(q75 - q25)


def _cpu_model():
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _versions(librosa, torch):
    import scipy

    parts = [f'Python {platform.python_version()}', f'afx-audio {afx_audio.__version__}',
             f'NumPy {np.__version__}', f'SciPy {scipy.__version__}']
    if librosa is not None:
        parts.append(f'librosa {librosa.__version__}')
    if torch is not None:
        parts.append(f'torch {torch.__version__} ({torch.get_num_threads()} threads)')
    return ', '.join(parts)


def _torch_times(torch, names, config, signal, batch, repeat):
    """Feature name -> ms per batch on the CPU for torch-supported features; 'all' is one module for all of them."""
    from afx_audio.torch import AFXFeatureModule, supported_features

    sr = config['sample_rate']
    waves = torch.from_numpy(np.tile(signal, (batch, 1))).to(torch.float32)
    supported = [n for n in names if n in supported_features()]
    out = {}
    with torch.no_grad():
        for name in supported:
            module = AFXFeatureModule({name: config['features'].get(name, {})}, sample_rate=sr)
            out[name] = _time(lambda m=module: m(waves), repeat)
        if supported:
            module = AFXFeatureModule({n: config['features'].get(n, {}) for n in supported}, sample_rate=sr)
            out['all'] = _time(lambda: module(waves), repeat)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(os.path.dirname(afx_audio.__file__), 'config.json'))
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--features', help='comma-separated subset')
    parser.add_argument('--librosa', action='store_true', help='also time the librosa counterparts')
    parser.add_argument('--torch-batch', type=int, default=0, help='also time the torch path on this many clips')
    parser.add_argument('--markdown', action='store_true', help='print a markdown table')
    args = parser.parse_args()

    librosa = torch = None
    if args.librosa:
        import librosa
    if args.torch_batch:
        import torch

    config = afx_audio.load_config(args.config)
    sr = config['sample_rate']
    rng = np.random.default_rng(0)
    t = np.arange(int(args.seconds * sr)) / sr
    signal = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.1 * rng.standard_normal(len(t))).astype(np.float32)

    names = args.features.split(',') if args.features else list(config['features'])
    lib_calls = _librosa_calls(librosa) if librosa is not None else {}
    torch_ms = _torch_times(torch, names, config, signal, args.torch_batch, args.repeat) if torch is not None else {}

    rows = []
    for name in names:
        params = config['features'].get(name, {})
        func = afx_audio.FEATURE_MAP[name]
        afx_ms = _time(lambda f=func, p=params: f(signal, sr, **p), args.repeat, afx_audio.clear_stft_cache)
        lib = None
        # librosa's frame-wise RMS has no whole-clip counterpart
        if name in lib_calls and not (name == 'rms_energy' and _param(name, params, 'frame_size') is None):
            label, call = lib_calls[name]
            lib = (label, _time(lambda c=call, n=name, p=params: c(signal, sr, n, p), args.repeat))
        rows.append((name, afx_ms, lib, torch_ms.get(name)))

    total = sum(r[1][0] for r in rows)
    run_config = {**config, 'features': {n: config['features'].get(n, {}) for n in names}}
    shared_ms, _ = _time(lambda: afx_audio.extract_all_features(signal, sr, run_config), args.repeat,
                         afx_audio.clear_stft_cache)
    if not args.markdown:
        for name, (ms, iqr), lib, tms in rows:
            line = f'{name:28s} {ms:9.2f} ms (IQR {iqr:.2f})'
            if lib:
                line += f'   librosa {lib[1][0]:9.2f} ms'
            if tms:
                line += f'   torch x{args.torch_batch} {tms[0]:9.2f} ms'
            print(line)
        print(f'{"total":28s} {total:9.2f} ms  ({args.seconds:g} s of audio at {sr} Hz)')
        print(f'{"extract_all_features":28s} {shared_ms:9.2f} ms  (shared STFT cache)')
        if 'all' in torch_ms:
            print(f'torch, all supported features in one module: {torch_ms["all"][0]:.2f} ms per batch of '
                  f'{args.torch_batch}')
        return

    print(f'CPU: {_cpu_model()}, {os.cpu_count()} logical cores, {platform.system()} {platform.release()}  ')
    print(f'{_versions(librosa, torch)}  ')
    print(f'Clip: {args.seconds:g} s at {sr} Hz, median of {args.repeat} runs after one warm-up (IQR in ms)')
    print()
    header = ['Feature', 'AFX ms', 'IQR']
    if librosa is not None:
        header += ['librosa ms', 'librosa function']
    if torch is not None:
        header += [f'torch ms per clip (batch of {args.torch_batch})']
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '|'.join(['---'] + ['---:'] * 2 + (['---:', '---'] if librosa is not None else [])
                         + (['---:'] if torch is not None else [])) + '|')
    for name, (ms, iqr), lib, tms in rows:
        cells = [f'`{name}`', f'{ms:.2f}', f'{iqr:.2f}']
        if librosa is not None:
            cells += [f'{lib[1][0]:.2f}', f'`{lib[0]}`'] if lib else ['', '']
        if torch is not None:
            cells += [f'{tms[0] / args.torch_batch:.2f}' if tms else '']
        print('| ' + ' | '.join(cells) + ' |')
    cells = ['total', f'{total:.2f}', '']
    if librosa is not None:
        cells += ['', '']
    if torch is not None:
        cells += [f'{torch_ms["all"][0] / args.torch_batch:.2f} (one module)' if 'all' in torch_ms else '']
    print('| ' + ' | '.join(cells) + ' |')
    print()
    print(f'`extract_all_features` on the same config, sharing one STFT per framing: {shared_ms:.2f} ms')


if __name__ == '__main__':
    main()
