import os

import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.exceptions import NotFittedError

os.environ["SAM_ML_LOG_LEVEL"] = "debug"
from sam_ml.models.main_regressor import SMAC_INSTALLED
from sam_ml.models.regressor import BYR, DTR, EN, ETR, LLCV, RFR, SGDR


def get_models() -> list:
    return [RFR(), DTR(), ETR(), SGDR(), LLCV(), EN(), BYR()]

X, Y = make_regression(n_samples = 50,
                        n_features = 5,
                        n_informative = 5,
                        random_state=42)
X = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4", "col5"])
Y = pd.Series(Y)


def test_regressor_fit_evaluate():
    for regressor in get_models():
        regressor.fit(X, Y)
        regressor.evaluate(X, Y, console_out=False)

def test_evaluate_score_error():
    with pytest.raises(NotFittedError):
        for regressor in get_models():
            regressor.evaluate_score(X,Y)

def test_evaluate_score_score():
    for regressor in get_models():
        regressor.train(X, Y)
        regressor.evaluate_score(X,Y)

def test_regressor_train_evaluate():
    for regressor in get_models():
        regressor.train(X, Y, console_out=False)
        regressor.evaluate(X, Y, console_out=False)
        regressor.evaluate_score(X, Y)

def test_regressor_crossvalidation():
    for regressor in get_models():
        regressor.cross_validation(X, Y, cv_num=2)

def test_regressor_crossvalidation_small_data():
    for regressor in get_models():
        regressor.cross_validation_small_data(X, Y)

def test_regressor_randomCVsearch():
    for regressor in get_models():
        best_param, _ = regressor.randomCVsearch(X, Y, n_trails=5, cv_num=2)
        assert best_param != {}, "should always find a parameter combination"

def test_regressor_randomCVsearch_small_data():
    for regressor in get_models():
        best_param, _ = regressor.randomCVsearch(X, Y, n_trails=5, cv_num=2, small_data_eval=True)
        assert best_param != {}, "should always find a parameter combination"

@pytest.mark.with_swig
def test_regressor_smac_search():
    if SMAC_INSTALLED:
        for regressor in get_models():
            best_param = regressor.smac_search(X, Y, n_trails=5, cv_num=2)
            assert best_param != {}, "should always find a parameter combination"
    else:
        with pytest.raises(ImportError):
            for regressor in get_models():
                best_param = regressor.smac_search(X, Y, n_trails=5, cv_num=2)
                assert best_param != {}, "should always find a parameter combination"

@pytest.mark.with_swig
def test_regressor_smac_search_small_data():
    if SMAC_INSTALLED:
        for regressor in get_models():
            best_param = regressor.smac_search(X, Y, n_trails=5, cv_num=2, small_data_eval=True)
            assert best_param != {}, "should always find a parameter combination"
    else:
        with pytest.raises(ImportError):
            for regressor in get_models():
                best_param = regressor.smac_search(X, Y, n_trails=5, cv_num=2, small_data_eval=True)
                assert best_param != {}, "should always find a parameter combination"
