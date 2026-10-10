# Examples

`example_extract_and_plot.py AUDIO`: extract the bundled config from one file, print the aggregated shapes and plot the MFCCs.

`classify_folders.py DATASET --preset speech`: cross-validated classification of a folder with one subfolder per class, through `AFXFeatureExtractor`, an imputer, a scaler and logistic regression.

Both are run by `tests/test_examples.py`.
