import pickle
import pandas as pd
import logging
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from .main_model import Model

class Classifier(Model):

    def __init__(self, model_object = None):
        self.model = model_object

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series, console_out: bool = True):
        logging.debug("evaluation started...")
        pred = self.model.predict(x_test)

        if len(y_test.unique()) == 2:
            avg = None
        else:
            avg = "micro"

        # Calculate Accuracy, Precision and Recall Metrics
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average=avg)
        recall = recall_score(y_test, pred, average=avg)

        if console_out:
            print("accuracy: ", accuracy)
            print("precision: ", precision)
            print("recall: ", recall)
        
            print("classification report: ")
            print(classification_report(y_test, pred))

        score = {
            "accuracy" : accuracy,
            "precision" : precision,
            "recall" : recall,
        }

        logging.debug("... evaluation finished")
        return score
    