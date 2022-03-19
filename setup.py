from setuptools import setup, find_packages


setup(
    name="sam_ml",
    version="0.1.0",
    description="a library for ML programing created by Samuel Brinkmann",
    packages=find_packages(),
    package_data={},
    scripts=[],
    install_requires=["scikit-learn", "pickle"],
    extras_require={
        "test": ["pytest", "pylint!=2.5.0"],
    },
    author='Samuel Brinkmann',
    license='MIT',
    tests_require=['pytest==4.4.1'],
    setup_requires=["pytest-runner"],
)
