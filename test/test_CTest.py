import os

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

os.environ["SAM_ML_SOUND_ON"] = "False"
from sam_ml.models import CTest

X, Y = make_classification(n_samples = 50,
                            n_features = 5,
                            n_informative = 5,
                            n_redundant = 0,
                            n_classes = 3,
                            weights = [.2, .3, .8])
X = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4", "col5"])
Y = pd.Series(Y)
x_train, x_test, y_train, y_test = train_test_split(X,Y, train_size=0.80, random_state=42)

def test_eval_models():
    ctest = CTest()
    ctest.eval_models(x_train, y_train, x_test, y_test)

def test_eval_models_cv():
    ctest = CTest()
    ctest.eval_models_cv(X, Y)

def test_eval_models_cv_small_data():
    ctest = CTest()
    ctest.eval_models_cv(X, Y, small_data_eval=True)

def test_find_best_model_randomCV():
    ctest = CTest()
    ctest.find_best_model_randomCV(x_train, y_train, x_test, y_test)

def test_find_best_model_randomCV_small_data():
    ctest = CTest()
    ctest.find_best_model_randomCV(x_train, y_train, x_test, y_test, small_data_eval=True)

def test_find_best_model_mass_search_error():
    with pytest.raises(RuntimeError):
        ctest = CTest()
        ctest.find_best_model_mass_search(x_train, y_train, x_test, y_test)

X_new, Y_new = make_classification(n_samples = 2000,
                            n_features = 5,
                            n_informative = 5,
                            n_redundant = 0,
                            n_classes = 3,
                            weights = [.2, .3, .8])
X_new = pd.DataFrame(X_new, columns=["col1", "col2", "col3", "col4", "col5"])
Y_new = pd.Series(Y_new)
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(X_new,Y_new, train_size=0.80, random_state=42)

def test_find_best_model_mass_search():
        ctest = CTest()
        ctest.find_best_model_mass_search(x_train_new, y_train_new, x_test_new, y_test_new, n_trails=3)
