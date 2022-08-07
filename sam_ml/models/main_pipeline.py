from typing import Union

from imblearn.pipeline import Pipeline as imbPipeline

from sam_ml.data import Embeddings_builder, Sampler, Scaler, Selector

from .main_classifier import Classifier


class Pipe(Classifier):
    """ pipeline Wrapper class """

    def __init__(self, components: list[Union[Embeddings_builder, Scaler, Selector, Sampler, Classifier]], model_name: str = "pipe"):
        """
        @params:
            components: list of sam_ml library Wrapper classes and only the last one is a subclass of Classifier
            model_name: name of the model
        """
        steps_list = []
        model_type = None
        grid = {}
        

        for component in components:
            if type(component) == Embeddings_builder:
                steps_list.append((component.vec_type, component.vectorizer))
            elif type(component) == Scaler:
                steps_list.append((component.scaler_type, component.scaler))
            elif type(component) == Selector:
                steps_list.append((component.algorithm, component.selector))
            elif type(component) == Sampler:
                steps_list.append((component.algorithm, component.sampler))
            elif issubclass(type(component), Classifier):
                steps_list.append((component.model_name, component.model))
                model_type = component.model_type
                pre_grid = {f"{component.model_name}__{k}": v for k, v in component.grid.items()}
                grid.update(pre_grid)

        model = imbPipeline(steps_list)
        super().__init__(model, model_name, model_type, grid)
