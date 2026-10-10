"""
Command-line interface for batch feature extraction.

Exit status: 0 on success, 1 when no file was found or every file failed,
2 for usage and configuration errors.
"""
import argparse
import os
import sys
import warnings

from afx_audio import __version__
from afx_audio.batch_extractor import FORMATS, extract_features_from_folder
from afx_audio.extract_all import FEATURE_MAP
from afx_audio.presets import list_presets, load_preset
from afx_audio.registry import feature_info, list_features
from afx_audio.utils.config_loader import VALID_AGGREGATIONS, unknown_feature_error

# Resolve the bundled default config relative to the installed package,
# not the current working directory.
DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'config.json'
)

# sliding_window needs window sizes that only a config file can give
_CLI_AGGREGATIONS = tuple(a for a in VALID_AGGREGATIONS if a != 'sliding_window')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='afx',
        description='Batch audio feature extraction over a folder of audio files.',
    )
    parser.add_argument('input_folder', nargs='?', help='Folder with audio files (searched recursively)')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--config', default=None, help='Path to a config JSON (default: bundled afx_audio/config.json)')
    source.add_argument('--preset', choices=list_presets(), help='Use a bundled domain preset instead of a config file')
    parser.add_argument('--output', '-o', default='features.npy',
                        help='Output file; .npy, .json, .csv or .parquet sets the format (default: features.npy)')
    parser.add_argument('--format', choices=FORMATS, default=None,
                        help='Output format, when the --output extension does not say it')
    parser.add_argument('--features', help='Comma-separated features to extract (default: all in the config)')
    parser.add_argument('--aggregation', choices=_CLI_AGGREGATIONS, help='Aggregation method over frames')
    parser.add_argument('--normalize', choices=['zscore', 'minmax'], help='Per-file normalization method')
    parser.add_argument('--no-flatten', action='store_true', help='Disable flattening after aggregation')
    parser.add_argument('--sample-rate', type=int, help='Resample audio to this rate (overrides the config)')
    parser.add_argument('--jobs', '-j', type=int, default=1,
                        help='Number of parallel worker processes (0 = all cores, default 1 = serial)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress per-file progress output')
    parser.add_argument('--list-features', action='store_true',
                        help='Print the available features with their group and units and exit')
    parser.add_argument('--describe', metavar='FEATURE',
                        help='Print the definition and parameters of a feature and exit')
    parser.add_argument('--list-presets', action='store_true', help='Print the bundled presets and exit')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    return parser


def _describe(name: str) -> None:
    info = feature_info(name)
    print(f"{info['name']} ({info['group']}, {info['units']}, rows per frame: {info['rows']})")
    print(info['summary'].replace('``', '`'))
    for key, value in info['parameters'].items():
        print(f'  {key} = {value!r}')


def _run(args, parser) -> int:
    if args.list_features:
        width = max(map(len, FEATURE_MAP))
        for name in list_features():
            info = feature_info(name)
            print(f"{name:<{width}}  {info['group']:<17}  {info['units']}")
        return 0
    if args.describe:
        if args.describe not in FEATURE_MAP:
            parser.error(str(unknown_feature_error(args.describe)))
        _describe(args.describe)
        return 0
    if args.list_presets:
        print('\n'.join(list_presets()))
        return 0
    if args.input_folder is None:
        parser.error('input_folder is required')
    if not os.path.isdir(args.input_folder):
        parser.error(f'input folder not found: {args.input_folder}')

    config: dict | str = load_preset(args.preset) if args.preset else (args.config or DEFAULT_CONFIG)
    feature_list = None
    if args.features is not None:
        feature_list = [f.strip() for f in args.features.split(',') if f.strip()]
        if not feature_list:
            parser.error('--features needs at least one feature name')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            results = extract_features_from_folder(
                args.input_folder,
                config,
                args.output,
                save_format=args.format,
                features=feature_list,
                aggregation=args.aggregation,
                flatten=False if args.no_flatten else None,
                normalize=args.normalize,
                n_jobs=args.jobs,
                sample_rate=args.sample_rate,
                verbose=not args.quiet,
            )
        except (ValueError, OSError, ImportError) as e:
            parser.exit(2, f'afx: error: {e}\n')
    for w in caught:
        print(f'afx: {w.message}', file=sys.stderr)

    errors = results.get('errors', {})
    n_ok = len(results) - (1 if errors else 0)
    if n_ok == 0:
        print('afx: no features were extracted', file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args, parser)
    except BrokenPipeError:
        # Output piped into a program that stopped reading (afx --list-features | head)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 0


if __name__ == '__main__':
    sys.exit(main())
