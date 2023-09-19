from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="sam_ml-py",
    version="1.0.0",
    description="a library for ML programing created by Samuel Brinkmann",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(),
    package_data={
        "sam_ml": [
            "data/regio/*.csv",
            "data/regio/*.dbf",
            "data/regio/*.prj",
            "data/regio/*.shp",
            "data/regio/*.shx",
            "models/microwave_finish_sound.mp3",
        ]
    },
    scripts=[],
    install_requires=[
        "scikit-learn<1.3",  # version 1.3 has some issues currently (04/07/2023)
        "pandas",
        "matplotlib",
        "numpy",
        "imbalanced-learn",
        "pygame",
        "ipywidgets",
        "tqdm",
        "statsmodels",
        "sentence-transformers",
        "xgboost",
        "ConfigSpace",  # for hyperparameter tuning spaces
        "geopandas",  # for regio data (e.g. germany map plot)
    ],
    extras_require={
        "test": ["pytest", "pylint", "isort", "refurb", "black"],
        "with_swig": ["smac"],
    },
    author="Samuel Brinkmann",
    license="MIT",
    tests_require=["pytest"],
    setup_requires=["pytest-runner"],
)
