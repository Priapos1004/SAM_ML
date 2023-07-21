import inspect


class DATA:

    def __init__(self, algorithm: str,  transformer: any):
        self.algorithm = algorithm
        self.transformer = transformer

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
        params = self.get_params(deep=False)
        init_params = inspect.signature(self.__init__).parameters
        init_params = {name: param.default for name, param in init_params.items()}

        init_params_transformer = inspect.signature(self.transformer.__init__).parameters
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
    
    def get_params(self, deep: bool = True):
        return {"algorithm": self.algorithm} | self.transformer.get_params(deep)

    def set_params(self, **params):
        self.transformer.set_params(**params)
        return self