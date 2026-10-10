"""
Extract features from one audio file with the bundled config, print the
aggregated feature vector, and plot the per-frame MFCCs.

Usage: python examples/example_extract_and_plot.py <audio_file> [--no-plot]
"""
import os
import sys

import afx_audio as afx
from afx_audio.utils.visualization import plot_mfcc

CONFIG_PATH = os.path.join(os.path.dirname(afx.__file__), 'config.json')


def main(argv):
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__.strip())
        return 1
    audio_path = argv[0]
    show_plot = '--no-plot' not in argv

    config = afx.load_config(CONFIG_PATH)
    signal, sr = afx.load_audio(audio_path, sr=config['sample_rate'])

    # File-level vector: mean over frames, one entry per feature dimension
    aggregated = afx.extract_all_features(signal, sr, config)
    for name, values in aggregated.items():
        print(f'{name:28s} shape={values.shape}')

    # Per-frame arrays for plotting
    config['preserve_shape'] = True
    per_frame = afx.extract_all_features(signal, sr, config)
    mfcc = per_frame['mfcc']
    print(f'mfcc per-frame shape: {mfcc.shape}')

    if show_plot:
        hop_length = config['features']['mfcc'].get('hop_length', 512)
        plot_mfcc(mfcc, sr, hop_length=hop_length)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
