"""
Command-line interface for batch feature extraction.
"""
import argparse
from AFX.batch_extractor import extract_features_from_folder

def main():
    parser = argparse.ArgumentParser(description='Batch audio feature extraction')
    parser.add_argument('input_folder', type=str, help='Folder with audio files')
    parser.add_argument('--config', type=str, default='audio_features/config.json', help='Path to config.json')
    parser.add_argument('--output', type=str, default='features.npy', help='Output file path (.npy, .json, .csv)')
    parser.add_argument('--format', type=str, default='npy', choices=['npy', 'json', 'csv'], help='Output format')
    args = parser.parse_args()
    extract_features_from_folder(args.input_folder, args.config, args.output, save_format=args.format)

if __name__ == '__main__':
    main()
