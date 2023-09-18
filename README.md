# SAM_ML

[![PyPI version](https://badge.fury.io/py/sam-ml-py.svg)](https://badge.fury.io/py/sam-ml-py)

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

--> in the '[examples](https://github.com/Priapos1004/SAM_ML/tree/main/examples)' folder you can find notebooks with code snippets that explain the usage of the package and it is also worth to look into the [wiki](https://github.com/Priapos1004/SAM_ML/wiki) of the repo.
