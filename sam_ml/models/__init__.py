from .AdaBoostClassifier import ABC
from .CatBoostClassifier import CBC
from .ClassifierTest import CTest
from .DecisionTreeClassifier import DTC
from .GradientBoostingMachine import GBM
from .KNeighborsClassifier import KNC
from .LogisticRegression import LR
from .main_classifier import Classifier
from .MLPClassifier import MLPC
from .RandomForestClassifier import RFC
from .SupportVectorClassifier import SVC

__all__ = {
    "main_classifier": "Classifier",
    "Classifier Testing": "CTest",
    "RandomForestClassifier": "RFC",
    "LogisticRegression": "LR",
    "DecisionTreeClassifier": "DCT",
    "SupportVectorClassifier": "SVC",
    "MLP Classifier": "MLPC",
    "GradientBoostingMachine": "GBM",
    "CatBoostClassifier": "CBC",
    "AdaBoostClassifier": "ABC",
    "KNeighborsClassifier": "KNC",
}
