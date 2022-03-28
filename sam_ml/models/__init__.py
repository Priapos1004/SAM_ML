from .main_classifier import Classifier
from .RandomForestClassifier import RFC
from .LogisticRegression import LR
from .DecisionTreeClassifier import DTC
from .SupportVectorClassifier import SVC
from .MLPClassifier import MLPC

__all__ = {
    "main_classifier": "Classifier",
    "RandomForestClassifier": "RFC",
    "LogisticRegression": "LR",
    "DecisionTreeClassifier": "DCT",
    "SupportVectorClassifier": "SVC",
    "MLP Classifier": "MLPC",
}
