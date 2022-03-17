from setuptools import setup, find_packages


setup(
    name="SAM_ML",
    version=0.1,
    description="a library for ML programing created by Samuel Brinkmann",
    packages=find_packages(),
    package_data={},
    scripts=[],
    install_requires=[
        "pandas",
    ],
    extras_require={
        "test": ["pytest", "pylint!=2.5.0"],
    },
    entry_points={
        "console_scripts": [],
    },
    classifiers=[],
    tests_require=["pytest"],
    setup_requires=["pytest-runner"],
    keywords="",
)
