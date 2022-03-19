import pickle
import logging
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

class classifier:

    def __init__(self, model_object):
        self.model = model_object

    def train(self, x_train, y_train):
        logging.debug("training started...")
        self.model.fit(x_train, y_train)
        logging.debug("... training finished")

    def evaluate(self, x_test, y_test, console_out: bool = True):
        logging.debug("evaluation started...")
        pred = self.model.predict(x_test)

        # Calculate Accuracy, Precision and Recall Metrics
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)

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

    def save(self, path: str):
        logging.debug("saving started...")
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logging.debug("... model saved")

    def load(self, path: str):
        logging.debug("loading model...")
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        logging.debug("... model loaded")
    