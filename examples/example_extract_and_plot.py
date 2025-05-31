"""
Example script: Extract features from a single audio file and plot MFCCs.
"""
import sys
from AFX.io.io import load_audio
from AFX.utils.config_loader import load_config
from AFX.extract_all import extract_all_features
from AFX.utils.visualization import plot_mfcc

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python example_extract_and_plot.py <audio_file>')
        sys.exit(1)
    audio_path = sys.argv[1]
    config = load_config('AFX/config.json')
    signal, sr = load_audio(audio_path, sr=config['sample_rate'])
    features = extract_all_features(signal, sr, config)
    if 'mfcc' in features:
        plot_mfcc(features['mfcc'], sr)
    else:
        print('MFCC not found in extracted features.')
