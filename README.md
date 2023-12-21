# SAM_ML

[![PyPI version](https://badge.fury.io/py/sam-ml-py.svg)](https://badge.fury.io/py/sam-ml-py)
[![docs](https://github.com/priapos1004/SAM_ML/workflows/docs/badge.svg)](https://priapos1004.github.io/SAM_ML/)
[![Code Quality Checks](https://github.com/priapos1004/SAM_ML/workflows/Code%20Quality%20Checks/badge.svg)](https://github.com/Priapos1004/SAM_ML/actions/workflows/CodeQualityChecks.yml)
[![view GitHub](https://img.shields.io/badge/View_on-GitHub-black?style=flat-square&logo=github)](https://github.com/Priapos1004/SAM_ML)

a library created by Samuel Brinkmann

## getting started (with SMAC3 library)

0. pre-installations *(needed for [smac](https://github.com/automl/SMAC3) library)*

You need to install `swig` for the *smag* library that is used for hyperparameter tuning.

Linux *(see [smac installation guide](https://automl.github.io/SMAC3/main/1_installation.html))*:

```
apt-get install swig
```

MacOS (with [homebrew](https://formulae.brew.sh/formula/swig)):

```
brew install swig
```

Windows (see [swigwin](https://www.swig.org/download.html))

1. install the package to your activated virtual environment *(python version 3.10 and higher is needed)*

```
pip install "sam-ml-py[with_swig]"
```

## getting started (without SMAC3 library)

1. install the package to your activated virtual environment *(python version 3.10 and higher is needed)*

```
pip install sam-ml-py
```

## start using it

2. now you can import the package, e.g.:

```
from sam_ml.models.classifier import RFC

RandomForestClassifier = RFC()
```

--> Look into the [docs](https://priapos1004.github.io/SAM_ML/) of the repo for more information.
