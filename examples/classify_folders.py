"""
Cross-validated classification of a folder of labelled audio.

    python examples/classify_folders.py DATASET [--preset speech] [--folds 5]

DATASET holds one subfolder per class (dog/*.wav, rain/*.wav, ...). Features
come from a preset through AFXFeatureExtractor; values that are undefined
for a clip (formants of noise, HNR of silence) are NaN and get the training
median, then a standardized logistic regression is scored with stratified
k-fold cross-validation.
"""
import argparse
import os
import sys

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import AFX
from AFX.sklearn import AFXFeatureExtractor

AUDIO_EXTS = ('.wav', '.flac', '.ogg', '.mp3')


def labelled_files(root):
    paths, labels = [], []
    for label in sorted(os.listdir(root)):
        folder = os.path.join(root, label)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(AUDIO_EXTS):
                paths.append(os.path.join(folder, name))
                labels.append(label)
    return paths, np.array(labels)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('dataset')
    parser.add_argument('--preset', default='speech', choices=AFX.list_presets())
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args(argv)

    paths, labels = labelled_files(args.dataset)
    if len(set(labels)) < 2:
        parser.error('need at least two class subfolders with audio files')
    config = AFX.load_preset(args.preset)
    model = make_pipeline(
        AFXFeatureExtractor(config=config, n_jobs=args.jobs),
        SimpleImputer(strategy='median'),
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )
    folds = min(args.folds, int(np.min(np.unique(labels, return_counts=True)[1])))
    scores = cross_val_score(model, paths, labels, cv=StratifiedKFold(folds, shuffle=True, random_state=0))
    print(f'{len(paths)} files, {len(set(labels))} classes, {folds}-fold accuracy '
          f'{scores.mean():.3f} +/- {scores.std():.3f}')
    return scores


if __name__ == '__main__':
    main(sys.argv[1:])
