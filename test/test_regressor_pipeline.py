import os

import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.exceptions import NotFittedError

os.environ["SAM_ML_LOG_LEVEL"] = "debug"
from sam_ml.models import Pipeline
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

def test_pipelines_fit_evaluate():
    for regressor in get_models():
        model = Pipeline(model=regressor, model_name=regressor.model_name)
        model.fit(X, Y)
        model.evaluate(X, Y, console_out=False)
        model.evaluate_score(X, Y)

def test_evaluate_score_error():
    with pytest.raises(NotFittedError):
        for regressor in get_models():
            model = Pipeline(model=regressor, model_name=regressor.model_name)
            model.evaluate_score(X, Y)

def test_evaluate_score():
    for regressor in get_models():
        model = Pipeline(model=regressor, model_name=regressor.model_name)
        model.train(X, Y)
        model.evaluate_score(X, Y)

def test_pipelines_train_warm_start():
    for regressor in get_models():
        model = Pipeline(model=regressor, model_name=regressor.model_name)
        assert model._data_classes_trained == False, "should be False with no training"
        model.train_warm_start(X, Y, console_out=False)
        assert model._data_classes_trained == True, "should be True after first training"
        model.train_warm_start(X, Y, console_out=False)
        assert model._data_classes_trained == True, "should still be True after second training"

def test_pipelines_train_warm_start():
    for regressor in get_models():
        model = Pipeline(model=regressor, model_name=regressor.model_name)
        assert model._data_classes_trained == False, "should be False with no training"
        model.fit_warm_start(X, Y)
        assert model._data_classes_trained == True, "should be True after first training"
        model.fit_warm_start(X, Y)
        assert model._data_classes_trained == True, "should still be True after second training"

def test_pipelines_train_evaluate():
    for regressor in get_models():
        model = Pipeline(model=regressor, model_name=regressor.model_name)
        model.train(X, Y, console_out=False)
        model.evaluate(X, Y, console_out=False)
        model.evaluate_score(X, Y)

def test_pipelines_crossvalidation():
    for regressor in get_models():
        model = Pipeline(model=regressor, model_name=regressor.model_name)
        model.cross_validation(X, Y, cv_num=2)

def test_pipelines_crossvalidation_small_data():
    for regressor in get_models():
        model = Pipeline(model=regressor, model_name=regressor.model_name)
        model.cross_validation_small_data(X, Y)

def test_pipelines_randomCVsearch():
    for regressor in get_models():
        model = Pipeline(model=regressor, model_name=regressor.model_name)
        best_param, _ = model.randomCVsearch(X, Y, n_trails=5, cv_num=2)
        assert best_param != {}, "should always find a parameter combination"

def test_pipelines_randomCVsearch_small_data():
    for regressor in get_models():
        model = Pipeline(model=regressor, model_name=regressor.model_name)
        best_param, _ = model.randomCVsearch(X, Y, n_trails=5, cv_num=2, small_data_eval=True)
        assert best_param != {}, "should always find a parameter combination"

@pytest.mark.with_swig
def test_pipelines_smac_search():
    if SMAC_INSTALLED:
        for regressor in get_models():
            model = Pipeline(model=regressor, model_name=regressor.model_name)
            best_param = model.smac_search(X, Y, n_trails=5, cv_num=2)
            assert best_param != {}, "should always find a parameter combination"
    else:
        with pytest.raises(ImportError):
            for regressor in get_models():
                model = Pipeline(model=regressor, model_name=regressor.model_name)
                best_param = model.smac_search(X, Y, n_trails=5, cv_num=2)
                assert best_param != {}, "should always find a parameter combination"

@pytest.mark.with_swig
def test_pipelines_smac_search_small_data():
    if SMAC_INSTALLED:
        for regressor in get_models():
            model = Pipeline(model=regressor, model_name=regressor.model_name)
            best_param = model.smac_search(X, Y, n_trails=5, cv_num=2, small_data_eval=True)
            assert best_param != {}, "should always find a parameter combination"
    else:
        with pytest.raises(ImportError):
            for regressor in get_models():
                model = Pipeline(model=regressor, model_name=regressor.model_name)
                best_param = model.smac_search(X, Y, n_trails=5, cv_num=2, small_data_eval=True)
                assert best_param != {}, "should always find a parameter combination"
