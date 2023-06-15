from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="sam_ml",
    version="0.3.0",
    description="a library for ML programing created by Samuel Brinkmann",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    packages=find_packages(),
    package_data={},
    scripts=[],
    install_requires=[
        "scikit-learn",
        "pandas",
        "matplotlib",
        "numpy",
        "catboost",
        "imbalanced-learn",
        "playsound",
        "PyObjC;platform_system=='Darwin'",
        "tqdm",
        "statsmodels",
        "azure-storage-blob",
    ],  # M1 problems with tensorflow, sentence-transformers, xgboost
    extras_require={"test": ["pytest", "pylint!=2.5.0", "isort", "refurb", "black"],},
    author="Samuel Brinkmann",
    license="MIT",
    tests_require=["pytest==4.4.1"],
    setup_requires=["pytest-runner"],
)
