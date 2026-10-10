# API

## Extraction

::: afx_audio.extract_all.extract_all_features

::: afx_audio.extract_all.extract_segments

::: afx_audio.registry.register_feature

::: afx_audio.registry.unregister_feature

::: afx_audio.batch_extractor.extract_features_from_folder

::: afx_audio.io.io.load_audio

## Configuration

::: afx_audio.utils.config_loader.load_config

::: afx_audio.utils.config_loader.validate_config

::: afx_audio.presets.load_preset

::: afx_audio.presets.resolve_preset

::: afx_audio.presets.list_presets

::: afx_audio.presets.preset_info

::: afx_audio.registry.feature_info

::: afx_audio.registry.list_features

::: afx_audio.utils.batch_utils.flatten_features

::: afx_audio.utils.batch_utils.feature_names

## Aggregation

::: afx_audio.utils.aggregator.aggregate_features

::: afx_audio.utils.aggregator.valid_fraction

## eGeMAPS

::: afx_audio.extractors.egemaps.egemaps

## Provenance and dataset statistics

::: afx_audio.provenance.config_hash

::: afx_audio.provenance.read_provenance

::: afx_audio.dataset_stats.load_stats

::: afx_audio.sklearn.scaler_from_stats

## Extractors

::: afx_audio.extractors.time_domain

::: afx_audio.extractors.frequency_domain

::: afx_audio.extractors.cepstral_features

::: afx_audio.extractors.harmonic_features

::: afx_audio.extractors.linear_prediction

## Signal processing building blocks

::: afx_audio.methods.stft.stft

::: afx_audio.methods.mel.mel_filterbank

::: afx_audio.methods.chroma.chroma_filterbank

::: afx_audio.methods.delta.compute_delta

::: afx_audio.methods.lpc.lpc

::: afx_audio.utils.pitch.yin

## scikit-learn

::: afx_audio.sklearn.AFXFeatureExtractor

## PyTorch

::: afx_audio.torch.module.AFXFeatureModule

## Warnings

Every warning AFX emits is an `AFXWarning`, so
`warnings.filterwarnings('ignore', category=afx.AFXWarning)` silences AFX
without hiding NumPy or SciPy warnings.

::: afx_audio._warnings.AFXWarning

::: afx_audio._warnings.AFXDeprecationWarning
