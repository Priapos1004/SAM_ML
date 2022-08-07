from .AdaBoostClassifier import ABC
from .BaggingClassifier import BC
from .BernoulliNB import BNB
from .CatBoostClassifier import CBC
from .ClassifierTest import CTest
from .DecisionTreeClassifier import DTC
from .ExtraTreesClassifier import ETC
from .GaussianNB import GNB
from .GaussianProcessClassifier import GPC
from .GradientBoostingMachine import GBM
from .KNeighborsClassifier import KNC
from .LinearDiscriminantAnalysis import LDA
from .LinearSupportVectorClassifier import LSVC
from .LogisticRegression import LR
from .main_classifier import Classifier
from .main_model import Model
from .main_pipeline import Pipe as Pipeline
from .MLPClassifier import MLPC
from .QuadraticDiscriminantAnalysis import QDA
from .RandomForestClassifier import RFC
from .SupportVectorClassifier import SVC

__all__ = {
    "model parent class": "Model",
    "classifier parent class": "Classifier",
    "pipeline class": "Pipeline",
    "AutoML class": "CTest",
    "RandomForestClassifier": "RFC",
    "LogisticRegression": "LR",
    "DecisionTreeClassifier": "DCT",
    "SupportVectorClassifier": "SVC",
    "MLP Classifier": "MLPC",
    "GradientBoostingMachine": "GBM",
    "CatBoostClassifier": "CBC",
    "AdaBoostClassifier": "ABC",
    "KNeighborsClassifier": "KNC",
    "ExtraTreesClassifier": "ETC",
    "GaussianNaiveBayes": "GNB",
    "BernoulliNaiveBayes": "BNB",
    "GaussianProcessClassifier": "GPC",
    "QuadraticDiscriminantAnalysis": "QDA",
    "LinearDiscriminantAnalysis": "LDA",
    "BaggingClassifier": "BC",
    "LinearSupportVectorClassifier": "LSVC",
}
