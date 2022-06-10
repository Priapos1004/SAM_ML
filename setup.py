from setuptools import find_packages, setup

setup(
    name="sam_ml",
    version="0.1.4",
    description="a library for ML programing created by Samuel Brinkmann",
    packages=find_packages(),
    package_data={},
    scripts=[],
    install_requires=["scikit-learn", "pandas", "matplotlib", "numpy", "catboost", "imbalanced-learn"], # M1 problems with tensorflow, sentence-transformers, xgboost
    extras_require={
        "test": ["pytest", "pylint!=2.5.0"],
    },
    author='Samuel Brinkmann',
    license='MIT',
    tests_require=['pytest==4.4.1'],
    setup_requires=["pytest-runner"],
)
