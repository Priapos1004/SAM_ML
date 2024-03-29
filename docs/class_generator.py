import inspect
from pathlib import Path
from typing import Callable

from jinja2 import Environment, FileSystemLoader

import sam_ml.data.preprocessing
import sam_ml.models.automl
import sam_ml.models.classifier
import sam_ml.models.regressor
from sam_ml.data.main_data import Data
from sam_ml.models.main_auto_ml import AutoML
from sam_ml.models.main_classifier import Classifier
from sam_ml.models.main_model import Model
from sam_ml.models.main_pipeline import BasePipeline
from sam_ml.models.main_regressor import Regressor


class PropertyInfo:
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.description = func.__doc__.replace("        Returns\n        -------\n", "") or 'No description available.'

class MethodInfo:
    def __init__(self, name: str, func: Callable, init: bool = False):
        self.name = name
        self.signature = inspect.signature(func)
        if init:
            self.description = func.__doc__.replace("        Parameters (important one)\n        --------------------------\n", "").replace("        Parameters\n        ----------\n", "").split("        Notes\n        -----\n")[0] or 'No description available.'
            notes_split = func.__doc__.split("        Notes\n        -----\n")
            if len(notes_split)==2:
                notes_header = """.. note::\n\n   """
                self.notes = notes_header+notes_split[1].strip()
            else:
                self.notes = ""
        else:
            self.description = inspect.getdoc(func) or 'No description available.'
        self.short_description = inspect.getdoc(func).split('\n')[0].strip() if inspect.getdoc(func) else 'No description available.'

class ClassInfo:
    def __init__(self, cls, abstract_class: bool, scoring_note: bool):
        if abstract_class:
            self.cls_name = f"{cls.__name__} class"
        else:
            self.cls_name = cls().model_name.split(" (")[0] + f" ({cls.__name__})" if hasattr(cls, "model_name") else cls.__name__

        self.full_cls_name = f"{cls.__module__}.{cls.__name__}"
        self.description = inspect.getdoc(cls) or 'No description available.'
        self.parent_class = cls.__base__.__name__

        if scoring_note or self.cls_name == "CTest":
            self.scoring_note =""".. note::\n\n   A lot of methods use parameters for advanced scoring. For additional information on advanced scoring, see :ref:`scoring documentation <classifier-scoring-section>`"""
        else:
            self.scoring_note = ""

        if abstract_class:
            self.example = ""
        else:
            self.example = f""".. raw:: html

   <h2>Example</h2>

>>> from {cls.__module__.rpartition(".")[0]} import {cls.__name__}
>>>
>>> model = {cls.__name__}()
>>> print(model)
{cls().__repr__()}
        """

def get_class_info(cls, abstract_class: bool, scoring_note: bool):
    cls_info = ClassInfo(cls, abstract_class=abstract_class, scoring_note=scoring_note)
    init_method = MethodInfo(cls.__name__, cls.__init__, init=True)
    if abstract_class:
        methods = [MethodInfo(func, getattr(cls, func)) for func in dir(cls) if callable(getattr(cls, func)) and not func.startswith("__")]
    else:
        methods = [MethodInfo(func, getattr(cls, func)) for func in dir(cls) if callable(getattr(cls, func)) and not func.startswith("_")]
    properties = [PropertyInfo(attr_name, getattr(cls, attr_name)) for attr_name in dir(cls) if isinstance(getattr(cls, attr_name), property)]
    return cls_info, methods, properties, init_method

def get_all_subclasses(cls):
    subclasses = set()
    queue = [cls]

    while queue:
        parent = queue.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                queue.append(child)

    return subclasses

def generate_folder(classes: list, folder_path: str, category_name: str, module_text: str = "", abstract_class: bool = False, scoring_note: bool = False):
    env = Environment(loader=FileSystemLoader('_templates'))
    template = env.get_template('class.rst')

    classes_names = []
    for cls in classes:
        cls_info, methods, properties, init_method = get_class_info(cls, abstract_class=abstract_class, scoring_note=scoring_note)
        classes_names.append(cls_info.cls_name)

        # If the folder does not exist, create it
        if not Path(folder_path).exists():
            Path(folder_path).mkdir(parents=True)

        Path(f'{folder_path}{cls_info.cls_name.replace(" ", "_")}.rst').write_text(
            template.render(
                cls_info=cls_info,
                methods=methods,
                properties=properties,
                init_method=init_method,
            ))
    
    if not abstract_class:
        # generate index.rst file
        with open(f"{folder_path}index.rst", "w") as f:
            f.write(f"{category_name}\n")
            f.write("="*len(category_name)+"\n\n")
            f.write(f"{module_text}\n\n")
            f.write(".. toctree::\n")
            f.write("   :maxdepth: 1\n\n")
            for name in sorted(classes_names):
                f.write(f"   {name.replace(' ', '_')}\n")

def main():
    # generate classifier folder
    classifier_text = "The ``sam_ml.models.classifier`` module contains several classifier wrapper models."
    generate_folder(get_all_subclasses(Classifier), "classifier/", "Classifier", classifier_text, scoring_note=True)
    # generate regressor folder
    regressor_text = "The ``sam_ml.models.regressor`` module contains several regressor wrapper models."
    generate_folder(get_all_subclasses(Regressor), "regressor/", "Regressor", regressor_text)
    # generate preprocessing folder
    preprocessing_text = "The ``sam_ml.data.preprocessing`` module contains several data preprocessing step wrapper transformers for different tasks, e.g. feature selection or sampling."
    generate_folder(get_all_subclasses(Data), "preprocessing/", "Preprocessing", preprocessing_text)
    # generate auto_ml folder
    automl_text = "The ``sam_ml.models.automl`` module contains two Auto-ML classes. One for Classifier (``CTest``) and one for Regressor (``RTest``)."
    generate_folder(get_all_subclasses(AutoML), "automl/", "Auto-ML", automl_text)
    # generate abstract_classes folder
    generate_folder([Model, Classifier, Regressor, BasePipeline, Data, AutoML], "abstract_classes/", "Abstract classes", abstract_class=True)

if __name__ == "__main__":
    main()
