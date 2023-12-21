import pandas as pd

from sam_ml.config import setup_logger

from ..main_data import Data
from .sampling import Sampler

logger = setup_logger(__name__)

class SamplerPipeline(Data):
    """ Class uses multplie up- and down-sampling algorithms instead of only one """

    def __init__(self, algorithm: str | list[Sampler] = "SMOTE_rus_20_50"):
        """
        Parameters
        ----------
        algorithm : str or list[Sampler], \
                default="SMOTE_rus_20_50"
            algorithm format:
            
            - "A1\_A2\_..._An_x1_x2_..._xn" with A1, A2, ... Sampler algorithm and x1, x2, ... their sampling_strategy
            
              first, use Sampler A1 with sampling_strategy x1% on data, then Sampler A2 with sampling_strategy x2% until Sampler An with sampling_strategy xn on data (ONLY works for binary data!!!)
              
              Note:
              sampling_strategy is the percentage of class size of minority in relation to the class size of the majority

              Examples (for binary classification):

              1) ros_rus_10_50: RandomOverSampler for minority class to 10% of majority class and then RandomUnderSampler for majority class to 2*minority class

              2) SMOTE_rus_20_50: SMOTE for minority class to 20% of majority class and then RandomUnderSampler for majority class to 2*minority class

            - list[Sampler]: use each Sampler in list one after the other on data
        """
        if type(algorithm) == str:
            sampler = self._build_sampling_pipeline(algorithm)
        else:
            sampler = algorithm
            algorithm = "custom"
        
        super().__init__(algorithm, sampler)
    
    def __repr__(self) -> str:
        return f"SamplerPipeline{tuple(self._transformer)}"
    
    def _build_sampling_pipeline(self, algorithm: str) -> list[Sampler]:
        """
        Function to create list of Samplers out of algorithm string

        Parameters
        ----------
        algorithm : str
            algorithm string in format "A1_A2_..._An_x1_x2_..._xn" with A1, A2, ... Sampler algorithm and x1, x2, ... their sampling_strategy

        Returns
        -------
        sampling_pipeline : list[Sampler]
            True if algorithm is valid
        """
        samplers_ratios = algorithm.split("_")
        if len(samplers_ratios)%2 == 1:
            raise ValueError(f"The string has to contain for every Sampler a sampling_strategy, but {samplers_ratios}")
        
        samplers = samplers_ratios[:int(len(samplers_ratios)/2)]
        ratios = samplers_ratios[int(len(samplers_ratios)/2):]
        ratios_float = [int(ratio)/100 for ratio in ratios]

        sampling_pipeline = []
        for idx in range(len(samplers)):
            if not (samplers[idx] in Sampler.params()["algorithm"] and 0<ratios_float[idx]<=1):
                raise ValueError(f"invalid sampler-sampling_strategy pair: '{samplers[idx]}' with {ratios_float[idx]}")
            else:
                sampling_pipeline.append(Sampler(algorithm=samplers[idx], sampling_strategy=ratios_float[idx]))

        return sampling_pipeline
    
    @staticmethod
    def params() -> dict:
        """
        Function to get the recommended parameter values for the class

        Returns
        -------
        param : dict
            recommended values for the parameter "algorithm"

        Examples
        --------
        >>> # get possible parameters
        >>> from sam_ml.data.preprocessing import SamplerPipeline
        >>>
        >>> # first way without class object
        >>> params1 = SamplerPipeline.params()
        >>> print(params1)
        {"algorithm": ["SMOTE_rus_20_50", ...]}
        >>> # second way with class object
        >>> model = SamplerPipeline()
        >>> params2 = model.params()
        >>> print(params2)
        {"algorithm": ["SMOTE_rus_20_50", ...]}
        """
        param = {"algorithm": ["SMOTE_rus_20_50", "ros_rus_10_50", "SMOTE_rus_20_100", "ros_rus_20_75"]}
        return param
    
    def get_params(self, deep: bool = True):
        return {"algorithm": self._transformer}

    def set_params(self, **params):
        if "algorithm" in params:
            algorithm = params["algorithm"]
            if type(algorithm) == str:
                self._transformer = self._build_sampling_pipeline(algorithm)
                self._algorithm = algorithm
            else:
                self._transformer = algorithm
                self._algorithm = "custom"
        else:
            raise ValueError("only the parameter 'algorithm' can be modified")

        return self

    def sample(self, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Function for up- and downsampling

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            data to sample

        Returns
        -------
        x_train_sampled : pd.DataFrame
            sampled x data
        y_train_sampled : pd.Series
            sampled y data

        Notes
        -----
        ONLY sample the `train data`. NEVER all data because then you will have some samples in train as well as in test data with random splitting

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_classification(n_samples=3000, n_features=4, n_classes=2, weights=[0.9], random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(y)
        >>> x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=42)
        >>> 
        >>> # sample data
        >>> from sam_ml.data.preprocessing import SamplerPipeline
        >>> model = SamplerPipeline()
        >>> x_train_sampled, y_train_sampled = model.sample(x_train, y_train)
        >>> print("before sampling:")
        >>> print(y_train.value_counts())
        >>> print()
        >>> print("after sampling:")
        >>> print(y_train_sampled.value_counts())
        before sampling:
        0    2140
        1     260
        Name: count, dtype: int64
        <BLANKLINE>
        after sampling:
        0    856
        1    428
        Name: count, dtype: int64
        """
        for sampler_idx in range(len(self._transformer)):
            if sampler_idx == 0:
                x_train_sampled, y_train_sampled = self._transformer[sampler_idx].sample(x_train, y_train)
            else:
                x_train_sampled, y_train_sampled = self._transformer[sampler_idx].sample(x_train_sampled, y_train_sampled)

        return x_train_sampled, y_train_sampled