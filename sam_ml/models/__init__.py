from .DecisionTreeClassifier import DTC
from .LogisticRegression import LR
from .main_classifier import Classifier
from .MLPClassifier import MLPC
from .RandomForestClassifier import RFC
from .SupportVectorClassifier import SVC

__all__ = {
    "main_classifier": "Classifier",
    "RandomForestClassifier": "RFC",
    "LogisticRegression": "LR",
    "DecisionTreeClassifier": "DCT",
    "SupportVectorClassifier": "SVC",
    "MLP Classifier": "MLPC",
}
