from setuptools import setup, find_packages

setup(
    name='audio_features',
    version='0.1.0',
    description='Audio feature extraction for ML/AI',
    author='Danilo Pena',
    packages=find_packages(),
    install_requires=[
        'librosa>=0.10.0',
        'numpy>=1.23.0',
        'scipy>=1.10.0',
        'matplotlib>=3.7.0',
        'pandas>=1.5.0',
        'seaborn>=0.12.0',
        'scikit-learn>=1.2.0',
        'soundfile>=0.12.0',
        'joblib>=1.2.0',
    ],
    python_requires='>=3.8',
    include_package_data=True,
)
