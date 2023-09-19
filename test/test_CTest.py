import os

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sam_ml.data.preprocessing import Sampler, Scaler, Selector
from sam_ml.models.main_classifier import SMAC_INSTALLED

os.environ["SAM_ML_SOUND_ON"] = "False"
os.environ["SAM_ML_LOG_LEVEL"] = "debug"
from sam_ml.models.classifier import CTest

X, Y = make_classification(n_samples = 50,
                            n_features = 5,
                            n_informative = 5,
                            n_redundant = 0,
                            n_classes = 3,
                            weights = [.2, .3, .8])
X = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4", "col5"])
Y = pd.Series(Y)
x_train, x_test, y_train, y_test = train_test_split(X,Y, train_size=0.80, random_state=42)


def test_eval_models_selectors():
    selectors = Selector.params()["algorithm"]
    ctest = CTest(models="all", selector=selectors)
    ctest.eval_models(x_train, y_train, x_test, y_test)

def test_eval_models_scalers():
    scalers = Scaler.params()["scaler"]
    ctest = CTest(models="all", scaler=scalers)
    ctest.eval_models(x_train, y_train, x_test, y_test)

def test_eval_models_cv():
    ctest = CTest(models="all")
    ctest.eval_models_cv(X, Y)

def test_eval_models_cv_small_data():
    ctest = CTest(models="all")
    ctest.eval_models_cv(X, Y, small_data_eval=True)

def test_find_best_model_randomCV():
    ctest = CTest(models="all")
    ctest.find_best_model_randomCV(x_train, y_train, x_test, y_test)

def test_find_best_model_randomCV_small_data():
    ctest = CTest(models="all")
    ctest.find_best_model_randomCV(x_train, y_train, x_test, y_test, small_data_eval=True)

@pytest.mark.with_swig
def test_find_best_model_smac():
    ctest = CTest(models="all")
    if SMAC_INSTALLED:
        ctest.find_best_model_smac(x_train, y_train, x_test, y_test)
    else:
        with pytest.raises(ImportError):
            ctest.find_best_model_smac(x_train, y_train, x_test, y_test)

@pytest.mark.with_swig
def test_find_best_model_smac_small_data():
    ctest = CTest(models="all")
    if SMAC_INSTALLED:
        ctest.find_best_model_smac(x_train, y_train, x_test, y_test, small_data_eval=True)
    else:
        with pytest.raises(ImportError):
            ctest.find_best_model_smac(x_train, y_train, x_test, y_test, small_data_eval=True)

def test_find_best_model_mass_search_error():
    with pytest.raises(RuntimeError):
        ctest = CTest(models="all")
        ctest.find_best_model_mass_search(x_train, y_train, x_test, y_test)

##### larger data set

X_200, Y_200 = make_classification(n_samples = 200,
                            n_features = 5,
                            n_informative = 5,
                            n_redundant = 0,
                            n_classes = 3,
                            weights = [.2, .3, .8])
X_200 = pd.DataFrame(X_200, columns=["col1", "col2", "col3", "col4", "col5"])
Y_200 = pd.Series(Y_200)
x_train_200, x_test_200, y_train_200, y_test_200 = train_test_split(X_200,Y_200, train_size=0.80, random_state=42)

X_5500, Y_5500 = make_classification(n_samples = 5500,
                            n_features = 5,
                            n_informative = 5,
                            n_redundant = 0,
                            n_classes = 3,
                            weights = [.2, .3, .8])
X_5500 = pd.DataFrame(X_5500, columns=["col1", "col2", "col3", "col4", "col5"])
Y_5500 = pd.Series(Y_5500)
x_train_5500, x_test_5500, y_train_5500, y_test_5500 = train_test_split(X_5500,Y_5500, train_size=0.80, random_state=42)


def test_eval_models_samplers():
    samplers = Sampler.params()["algorithm"]
    ctest = CTest(models="all", sampler=samplers)
    ctest.eval_models(x_train_200, y_train_200, x_test_200, y_test_200)

def test_find_best_model_mass_search():
    ctest = CTest(models="all")
    ctest.find_best_model_mass_search(x_train_5500, y_train_5500, x_test_5500, y_test_5500, n_trails=3)
