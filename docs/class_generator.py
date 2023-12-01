import inspect
from typing import Callable

from jinja2 import Environment, FileSystemLoader

import sam_ml.models.classifier
import sam_ml.models.regressor
from sam_ml.models.main_classifier import Classifier
from sam_ml.models.main_regressor import Regressor


class PropertyInfo:
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.description = func.__doc__ if func.__doc__ else 'No description available.'

class MethodInfo:
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.signature = inspect.signature(func)
        self.description = func.__doc__ if func.__doc__ else 'No description available.'
        self.short_description = func.__doc__.split('\n')[1].strip() if func.__doc__ else 'No description available.'

class ClassInfo:
    def __init__(self, cls):
        self.cls_name = cls().model_name.split(" ")[0]
        self.full_cls_name = f"{cls.__module__}.{cls.__name__}"
        self.description = cls.__doc__ if cls.__doc__ else 'No description available.'
        self.example = f"""
>>> # load data (replace with own data)
>>> X, y = ...
>>>
>>> from sam_ml.models.classifier import {cls.__name__}
>>>
>>> model = {cls.__name__}()
>>> print(model)
{cls().__repr__()}
        """

def get_class_info(cls):
    cls_info = ClassInfo(cls)
    init_method = MethodInfo(cls.__name__, cls.__init__)
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

def generate_folder(parent_class, folder_path: str, category_name: str):
    env = Environment(loader=FileSystemLoader('_templates'))
    template = env.get_template('class.rst')

    classes = get_all_subclasses(parent_class)  # List your classes here
    classes_names = []
    for cls in classes:
        cls_info, methods, properties, init_method = get_class_info(cls)
        classes_names.append(cls_info.cls_name)
        with open(f'{folder_path}{cls_info.cls_name}.rst', 'w') as f:
            f.write(template.render(
                cls_info=cls_info,
                methods=methods,
                properties=properties,
                init_method=init_method,
            ))
    
    # generate index.rst file
    with open(f"{folder_path}index.rst", "w") as f:
        f.write(f"{category_name}\n")
        f.write("==========\n\n")
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 1\n\n")
        for name in sorted(classes_names):
            f.write(f"   {name}\n")
    
    # update main index
    with open("index.rst", "a") as f:
        f.write(f"   {folder_path}index.rst")

def main():
    # generate index file
    index_content = """Welcome to sam-ml-py's documentation!
=====================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

"""
    with open("index.rst", "w") as f:
        f.write(index_content)

    # add files
    generate_folder(Classifier, "classifier/", "Classifier")
    # generate_folder(Regressor, "regressor/", "Regressor")

if __name__ == "__main__":
    main()
