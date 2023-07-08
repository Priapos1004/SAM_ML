import pandas as pd

from sam_ml.config import setup_logger

from .sampling import Sampler

logger = setup_logger(__name__)

class SamplerPipeline:
    def __init__(self, algorithm: str | list[Sampler] = "ros_rus_20_50"):
        """
        Class uses multplie up- and down-sampling algorithms instead of only one

        @param:
            - algorithm = "A1_A2_..._An_x1_x2_..._xn": first, use Sampler A1 with sampling_strategy x1% on data, then Sampler A2 with sampling_strategy x2% until Sampler An with sampling_strategy xn on data (only works for binary data!!!)
            - algorithm = list[Sampler]: use each Sampler in list one after the other on data

        @Note:
            - sampling_strategy is the percentage of minority class size of majority class size

        @example:
            - ros_rus_10_50: RandomOverSampler for minority class to 10% of majority class and then RandomUnderSampler for majority class to 2*minority class
            - SMOTE_rus_20_50: SMOTE for minority class to 20% of majority class and then RandomUnderSampler for majority class to 2*minority class
        """
        if type(algorithm) == str:
            self.algorithm = algorithm

            samplers_ratios = algorithm.split("_")
            if len(samplers_ratios)%2 == 1:
                raise ValueError(f"The string has to contain for every Sampler a sampling_strategy, but {samplers_ratios}")
            
            samplers = samplers_ratios[:int(len(samplers_ratios)/2)]
            ratios = samplers_ratios[int(len(samplers_ratios)/2):]
            ratios_float = [int(ratio)/100 for ratio in ratios]

            self.sampler = [Sampler(algorithm=samplers[idx], sampling_strategy=ratios_float[idx]) for idx in range(len(samplers))]
        else:
            self.sampler = algorithm
            self.algorithm = "custom"
    
    def __repr__(self) -> str:
        return f"SamplerPipeline{tuple(self.sampler)}"
    
    @staticmethod
    def check_is_valid_algorithm(algorithm: str) -> bool:
        """
        @return:
            True if algorithm is valid
        """
        samplers_ratios = algorithm.split("_")
        if len(samplers_ratios)%2 == 1:
            logger.warning(f"The string has to contain for every Sampler a sampling_strategy, but {samplers_ratios}")
            return False
        
        samplers = samplers_ratios[:int(len(samplers_ratios)/2)]
        ratios = samplers_ratios[int(len(samplers_ratios)/2):]
        ratios_float = [int(ratio)/100 for ratio in ratios]

        for idx in range(len(samplers)):
            if not (samplers[idx] in Sampler.params()["algorithm"] and 0<ratios_float[idx]<=1):
                logger.warning(f"invalid sampler-sampling_strategy pair: '{samplers[idx]}' with {ratios_float[idx]}")
                return False

        return True
    
    def get_params(self, deep: bool = True):
        return {"algorithm": self.sampler}

    def set_params(self, *params):
        self.sampler = params
        return self

    def sample(self, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Function for up- and downsampling

        @return:
            tuple x_train_sampled, y_train_sampled
        """
        for sampler_idx in range(len(self.sampler)):
            if sampler_idx == 0:
                x_train_sampled, y_train_sampled = self.sampler[sampler_idx].sample(x_train, y_train)
            else:
                x_train_sampled, y_train_sampled = self.sampler[sampler_idx].sample(x_train_sampled, y_train_sampled)

        return x_train_sampled, y_train_sampled