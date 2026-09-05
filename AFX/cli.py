"""
Command-line interface for batch feature extraction.
"""
import argparse
import os
from AFX.batch_extractor import extract_features_from_folder

# Resolve the bundled default config relative to the installed package,
# not the current working directory.
DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'config.json'
)

def main():
    parser = argparse.ArgumentParser(description='Batch audio feature extraction')
    parser.add_argument('input_folder', type=str, help='Folder with audio files')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG, help='Path to config.json')
    parser.add_argument('--output', type=str, default='features.npy', help='Output file path (.npy, .json, .csv, .parquet)')
    parser.add_argument('--format', type=str, default='npy', choices=['npy', 'json', 'csv', 'parquet'], help='Output format')

    parser.add_argument('--features', type=str, help='Comma-separated list of features to extract')
    parser.add_argument('--aggregation', type=str, help='Aggregation method to use')
    parser.add_argument('--normalize', type=str, choices=['zscore', 'minmax'], help='Normalization method')
    parser.add_argument('--no-flatten', action='store_true', help='Disable flattening after aggregation')
    parser.add_argument('--jobs', '-j', type=int, default=1,
                        help='Number of parallel worker processes '
                             '(0 = all cores, default 1 = serial)')

    args = parser.parse_args()

    feature_list = None
    if args.features:
        feature_list = [f.strip() for f in args.features.split(',') if f.strip()]
    extract_features_from_folder(
        args.input_folder,
        args.config,
        args.output,
        save_format=args.format,
        features=feature_list,
        aggregation=args.aggregation,
        flatten=False if args.no_flatten else None,
        normalize=args.normalize,
        n_jobs=args.jobs,
    )

if __name__ == '__main__':
    main()
