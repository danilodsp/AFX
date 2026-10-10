"""Time every feature in a config on synthetic audio.

    python benchmarks/bench_features.py [--config PATH] [--seconds 10] [--repeat 5]

Reports the median wall time per feature in milliseconds. The STFT cache is
cleared before each call, so every feature pays for its own transform.
"""
import argparse
import os
import time

import numpy as np

import AFX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(os.path.dirname(AFX.__file__), 'config.json'))
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--features', help='comma-separated subset')
    args = parser.parse_args()

    config = AFX.load_config(args.config)
    sr = config['sample_rate']
    rng = np.random.default_rng(0)
    t = np.arange(int(args.seconds * sr)) / sr
    signal = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.1 * rng.standard_normal(len(t))).astype(np.float32)

    names = args.features.split(',') if args.features else list(config['features'])
    total = 0.0
    for name in names:
        params = config['features'].get(name, {})
        times = []
        for _ in range(args.repeat):
            AFX.clear_stft_cache()
            t0 = time.perf_counter()
            AFX.FEATURE_MAP[name](signal, sr, **params)
            times.append(time.perf_counter() - t0)
        ms = 1000 * float(np.median(times))
        total += ms
        print(f'{name:28s} {ms:9.2f} ms')
    print(f'{"total":28s} {total:9.2f} ms  ({args.seconds:g} s of audio at {sr} Hz)')


if __name__ == '__main__':
    main()
