from sklearn.neural_network import MLPClassifier

from .main_classifier import Classifier


class MLPC(Classifier):
    """ MLP Classifier Wrapper class """

    def __init__(
        self,
        model_name: str = "MLP Classifier",
        random_state: int = 42,
        **kwargs,
    ):
        """
        @param (important one):
            hidden_layer_sizes: the ith element represents the number of neurons in the ith hidden layer
            activation: activation function for the hidden layer
            solver: solver for weight optimization
            alpha: l2 penalty (regularization term) parameter
            learning_rate: learning rate schedule for weight updates
            warm_start: work with previous fit and add more estimator
            tol: Tolerance for stopping criteria
            max_iter: Maximum number of iterations taken for the solvers to converge

            random_state: random_state for model
            verbose: logging (True/False)
            batch_size: Size of minibatches for stochastic optimizers
            early_stopping: True: tests on 10% of train data and stops if there is for 'n_iter_no_change' no improvement in the metrics
        """
        model_type = "MLPC"
        model = MLPClassifier(
            random_state=random_state,
            **kwargs,
        )
        grid = {
            "hidden_layer_sizes": [(10, 30, 10), (20,), (10,), (100,), (50,50,50), (50,100,50)],
            "activation": ["tanh", "relu", "logistic"],
            "solver": ["sgd", "adam"],
            "alpha": [0.0001, 0.001, 0.01, 0.05],
            "learning_rate": ["constant", "adaptive"],
        }
        super().__init__(model, model_name, model_type, grid)
