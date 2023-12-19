import inspect
from abc import abstractmethod


class Data:
    """ Data preprocessing parent class {abstract} """

    def __init__(self, algorithm: str,  transformer: any):
        """
        Parameters
        ----------
        algorithm : str
            name of the used algorithm
        transformer : transformer instance
            transformer instance (e.g. StandardScaler)
        """
        self._algorithm = algorithm
        self._transformer = transformer

    def __repr__(self) -> str:
        transformer_params: str = ""
        param_dict = self._changed_parameters()
        for key in param_dict:
            if type(param_dict[key]) == str:
                transformer_params += key+"='"+str(param_dict[key])+"', "
            else:
                transformer_params += key+"="+str(param_dict[key])+", "

        return f"{self.__class__.__name__}({transformer_params})"

    def _changed_parameters(self):
        """
        Returns
        -------
        dictionary of model parameter that are different from default values
        """
        params = self.get_params(deep=False)
        init_params = inspect.signature(self.__init__).parameters
        init_params = {name: param.default for name, param in init_params.items()}

        init_params_transformer = inspect.signature(self._transformer.__init__).parameters
        init_params_transformer = {name: param.default for name, param in init_params_transformer.items()}

        def has_changed(k, v):
            if k not in init_params:  # happens if k is part of a **kwargs
                if k not in init_params_transformer: # happens if k is part of a **kwargs
                    return True
                else:
                    if v != init_params_transformer[k]:
                        return True
                    else:
                        return False

            if init_params[k] == inspect._empty:  # k has no default value
                return True
            elif init_params[k] != v:
                return True
            
            return False

        return {k: v for k, v in params.items() if has_changed(k, v)}
    
    @property
    def algorithm(self) -> str:
        """
        Returns
        -------
        algorithm : str
            name of the used algorithm
        """
        return self._algorithm
    
    @property
    def transformer(self) -> any:
        """
        Returns
        -------
        transformer : transformer instance
            transformer instance (e.g. StandardScaler)
        """
        return self._transformer
    
    @staticmethod
    @abstractmethod
    def params() -> dict:
        """
        Function to get the possible/recommended parameter values for the class

        Returns
        -------
        param : dict
            possible/recommended values for the parameters
        """
        pass
    
    def get_params(self, deep: bool = True) -> dict:
        """
        Function to get the parameter from the transformer instance

        Parameters
        ----------
        deep : bool, \
                default=True
            If True, will return the parameters for this estimator and contained sub-objects that are estimators

        Returns
        -------
        params: dict
            parameter names mapped to their values
        """
        return {"algorithm": self.algorithm} | self.transformer.get_params(deep)

    def set_params(self, **params):
        """
        Function to set the parameter of the transformer instance

        Parameters
        ----------
        \*\*params : dict
            Estimator parameters

        Returns
        -------
        self : estimator instance
            Estimator instance
        """
        self._transformer.set_params(**params)
        return self