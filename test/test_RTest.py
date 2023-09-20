import os

import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from sam_ml.data.preprocessing import Sampler, Scaler, Selector
from sam_ml.models.main_classifier import SMAC_INSTALLED

os.environ["SAM_ML_SOUND_ON"] = "False"
os.environ["SAM_ML_LOG_LEVEL"] = "debug"
from sam_ml.models.regressor import RTest

X, Y = make_regression(n_samples = 50,
                        n_features = 5,
                        n_informative = 5,
                        random_state=42)
X = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4", "col5"])
Y = pd.Series(Y)
x_train, x_test, y_train, y_test = train_test_split(X,Y, train_size=0.80, random_state=42)


def test_eval_models_selectors():
    selectors = Selector.params()["algorithm"]
    rtest = RTest(models="all", selector=selectors)
    rtest.eval_models(x_train, y_train, x_test, y_test)

def test_eval_models_scalers():
    scalers = Scaler.params()["scaler"]
    rtest = RTest(models="all", scaler=scalers)
    rtest.eval_models(x_train, y_train, x_test, y_test)

def test_eval_models_cv():
    rtest = RTest(models="all")
    rtest.eval_models_cv(X, Y)

def test_eval_models_cv_small_data():
    rtest = RTest(models="all")
    rtest.eval_models_cv(X, Y, small_data_eval=True)

def test_find_best_model_randomCV():
    rtest = RTest(models="all")
    rtest.find_best_model_randomCV(x_train, y_train, x_test, y_test)

def test_find_best_model_randomCV_small_data():
    rtest = RTest(models="all")
    rtest.find_best_model_randomCV(x_train, y_train, x_test, y_test, small_data_eval=True)

@pytest.mark.with_swig
def test_find_best_model_smac():
    rtest = RTest(models="all")
    if SMAC_INSTALLED:
        rtest.find_best_model_smac(x_train, y_train, x_test, y_test)
    else:
        with pytest.raises(ImportError):
            rtest.find_best_model_smac(x_train, y_train, x_test, y_test)

@pytest.mark.with_swig
def test_find_best_model_smac_small_data():
    rtest = RTest(models="all")
    if SMAC_INSTALLED:
        rtest.find_best_model_smac(x_train, y_train, x_test, y_test, small_data_eval=True)
    else:
        with pytest.raises(ImportError):
            rtest.find_best_model_smac(x_train, y_train, x_test, y_test, small_data_eval=True)

def test_find_best_model_mass_search_error():
    with pytest.raises(RuntimeError):
        rtest = RTest(models="all")
        rtest.find_best_model_mass_search(x_train, y_train, x_test, y_test)

##### larger data set

X_200, Y_200 = make_regression(n_samples = 200,
                        n_features = 5,
                        n_informative = 5,
                        random_state=42)
X_200 = pd.DataFrame(X_200, columns=["col1", "col2", "col3", "col4", "col5"])
Y_200 = pd.Series(Y_200)
x_train_200, x_test_200, y_train_200, y_test_200 = train_test_split(X_200,Y_200, train_size=0.80, random_state=42)

X_5500, Y_5500 = make_regression(n_samples = 5500,
                        n_features = 5,
                        n_informative = 5,
                        random_state=42)
X_5500 = pd.DataFrame(X_5500, columns=["col1", "col2", "col3", "col4", "col5"])
Y_5500 = pd.Series(Y_5500)
x_train_5500, x_test_5500, y_train_5500, y_test_5500 = train_test_split(X_5500,Y_5500, train_size=0.80, random_state=42)


def test_eval_models_samplers():
    samplers = Sampler.params()["algorithm"]
    rtest = RTest(models="all", sampler=samplers)
    rtest.eval_models(x_train_200, y_train_200, x_test_200, y_test_200)

def test_find_best_model_mass_search():
    rtest = RTest(models="all")
    rtest.find_best_model_mass_search(x_train_5500, y_train_5500, x_test_5500, y_test_5500, n_trails=3)
