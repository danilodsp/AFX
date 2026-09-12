"""
Command-line interface for batch feature extraction.
"""
import argparse
import os
import sys

from AFX.batch_extractor import extract_features_from_folder
from AFX.extract_all import FEATURE_MAP
from AFX.presets import list_presets, load_preset
from AFX.utils.config_loader import VALID_AGGREGATIONS

# Resolve the bundled default config relative to the installed package,
# not the current working directory.
DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'config.json'
)

FORMATS = ('npy', 'json', 'csv', 'parquet')


def _format_from_output(output: str) -> str:
    ext = os.path.splitext(output)[1].lower().lstrip('.')
    return ext if ext in FORMATS else 'npy'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='afx',
        description='Batch audio feature extraction over a folder of audio files.',
    )
    parser.add_argument('input_folder', nargs='?', help='Folder with audio files (searched recursively)')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--config', default=None, help='Path to a config JSON (default: bundled AFX/config.json)')
    source.add_argument('--preset', choices=list_presets(), help='Use a bundled domain preset instead of a config file')
    parser.add_argument('--output', default='features.npy', help='Output file path (default: features.npy)')
    parser.add_argument('--format', choices=FORMATS, default=None,
                        help='Output format (default: inferred from --output extension, else npy)')
    parser.add_argument('--features', help='Comma-separated subset of the config features to extract')
    parser.add_argument('--aggregation', choices=VALID_AGGREGATIONS, help='Aggregation method over frames')
    parser.add_argument('--normalize', choices=['zscore', 'minmax'], help='Per-file normalization method')
    parser.add_argument('--no-flatten', action='store_true', help='Disable flattening after aggregation')
    parser.add_argument('--sample-rate', type=int, help='Resample audio to this rate (overrides the config)')
    parser.add_argument('--jobs', '-j', type=int, default=1,
                        help='Number of parallel worker processes (0 = all cores, default 1 = serial)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress per-file progress output')
    parser.add_argument('--list-features', action='store_true', help='Print the available feature names and exit')
    parser.add_argument('--list-presets', action='store_true', help='Print the bundled presets and exit')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_features:
        print('\n'.join(sorted(FEATURE_MAP)))
        return 0
    if args.list_presets:
        print('\n'.join(list_presets()))
        return 0
    if args.input_folder is None:
        parser.error('input_folder is required')
    if not os.path.isdir(args.input_folder):
        parser.error(f'input folder not found: {args.input_folder}')

    if args.preset:
        config = load_preset(args.preset)
    else:
        config = args.config or DEFAULT_CONFIG

    feature_list = None
    if args.features:
        feature_list = [f.strip() for f in args.features.split(',') if f.strip()]

    extract_features_from_folder(
        args.input_folder,
        config,
        args.output,
        save_format=args.format or _format_from_output(args.output),
        features=feature_list,
        aggregation=args.aggregation,
        flatten=False if args.no_flatten else None,
        normalize=args.normalize,
        n_jobs=args.jobs,
        sample_rate=args.sample_rate,
        verbose=not args.quiet,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
