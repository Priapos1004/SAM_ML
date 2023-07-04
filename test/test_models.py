import pandas as pd
from sklearn.datasets import make_classification

from sam_ml.models import (
    ABC,
    BC,
    BNB,
    DTC,
    ETC,
    GBM,
    GNB,
    GPC,
    KNC,
    LDA,
    LR,
    LSVC,
    MLPC,
    QDA,
    RFC,
    SVC,
    XGBC,
    Pipeline,
)

MODELS = [ABC(), BC(), BNB(), DTC(), ETC(), GNB(), GPC(), GBM(), KNC(), LDA(), LSVC(), LR(), MLPC(), QDA(), RFC(), SVC(), XGBC()]
X, Y = make_classification(n_samples = 50,
                            n_features = 5,
                            n_informative = 5,
                            n_redundant = 0,
                            n_classes = 3,
                            weights = [.2, .3, .8])
X = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4", "col5"])
Y = pd.Series(Y)


def test_classifier_train_evaluate():
    for classifier in MODELS:
        classifier.train(X, Y, console_out=False)
        classifier.evaluate(X, Y, console_out=False)
        classifier.evaluate_score(X, Y)


def test_pipelines_train_evaluate():
    for classifier in MODELS:
        model = Pipeline(model=classifier, model_name=classifier.model_name)
        model.train(X, Y, console_out=False)
        model.evaluate(X, Y, console_out=False)
        model.evaluate_score(X, Y)

def test_classifier_crossvalidation():
    for classifier in MODELS:
        classifier.cross_validation(X, Y, cv_num=3)

def test_pipelines_crossvalidation():
    for classifier in MODELS:
        model = Pipeline(model=classifier, model_name=classifier.model_name)
        model.cross_validation(X, Y, cv_num=3)

def test_classifier_crossvalidation_small_data():
    for classifier in MODELS:
        classifier.cross_validation_small_data(X, Y)

def test_pipelines_crossvalidation_small_data():
    for classifier in MODELS:
        model = Pipeline(model=classifier, model_name=classifier.model_name)
        model.cross_validation_small_data(X, Y)

def test_classifier_randomCVsearch():
    for classifier in MODELS:
        best_param, _ = classifier.randomCVsearch(X, Y, n_trails=5, cv_num=3)
        assert best_param != {}, "should always find a parameter combination"

def test_pipelines_randomCVsearch():
    for classifier in MODELS:
        model = Pipeline(model=classifier, model_name=classifier.model_name)
        best_param, _ = model.randomCVsearch(X, Y, n_trails=5, cv_num=3)
        assert best_param != {}, "should always find a parameter combination"

def test_classifier_smac_search():
    for classifier in MODELS:
        best_param = classifier.smac_search(X, Y, n_trails=10, cv_num=3)
        assert best_param != {}, "should always find a parameter combination"

def test_pipelines_smac_search():
    for classifier in MODELS:
        model = Pipeline(model=classifier, model_name=classifier.model_name)
        best_param = model.smac_search(X, Y, n_trails=10, cv_num=3)
        assert best_param != {}, "should always find a parameter combination"
