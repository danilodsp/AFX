"""
Command-line interface for batch feature extraction.
"""
import argparse
from AFX.batch_extractor import extract_features_from_folder

def main():
    parser = argparse.ArgumentParser(description='Batch audio feature extraction')
    parser.add_argument('input_folder', type=str, help='Folder with audio files')
    parser.add_argument('--config', type=str, default='AFX/config.json', help='Path to config.json')
    parser.add_argument('--output', type=str, default='features.npy', help='Output file path (.npy, .json, .csv)')
    parser.add_argument('--format', type=str, default='npy', choices=['npy', 'json', 'csv'], help='Output format')

    parser.add_argument('--features', type=str, help='Comma-separated list of features to extract')
    parser.add_argument('--aggregation', type=str, help='Aggregation method to use')
    parser.add_argument('--normalize', type=str, choices=['zscore', 'minmax'], help='Normalization method')
    parser.add_argument('--no-flatten', action='store_true', help='Disable flattening after aggregation')

    args = parser.parse_args()

    feature_list = args.features.split(',') if args.features else None
    extract_features_from_folder(
        args.input_folder,
        args.config,
        args.output,
        save_format=args.format,
        features=feature_list,
        aggregation=args.aggregation,
        flatten=not args.no_flatten,
        normalize=args.normalize,
    )

if __name__ == '__main__':
    main()
