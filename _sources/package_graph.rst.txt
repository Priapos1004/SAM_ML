Package Graph
=============

.. uml::

    @startuml
    skinparam classAttributeIconSize 0
    hide empty members

    package "sam_ml" {

        package "sam_ml.data" {
            abstract class Data
            package "sam_ml.data.preprocessing" {
                circle " "
            }
            package "sam_ml.data.regio" {
                circle "     "
            }
        }

        package "sam_ml.models" {
            abstract class Model
            abstract class Classifier
            abstract class Regressor
            abstract class BasePipeline
            protocol create_pipeline
            class DynamicPipeline
            package "sam_ml.models.classifier" {
                circle "  "
            }
            package "sam_ml.models.regressor" {
                circle "   "
            }

            abstract class AutoML
            package "sam_ml.models.automl" {
                circle "    "
            }
        }
    }

    AutoML <|-- "    "
    Data <|-- " "
    Model <|-- Classifier
    Model <|-- Regressor
    Model <|-- BasePipeline
    Classifier <|-- "  "
    Classifier <|-- DynamicPipeline
    Regressor <|-- DynamicPipeline
    Regressor <|-- "   "
    DynamicPipeline <-- create_pipeline
    BasePipeline <|-- DynamicPipeline

    @enduml

.. note::

    Class Factory :func:`create_pipeline <sam_ml.models.create_pipeline>`

    Creates **DynamicPipeline** class dynamically 
    based on input model (:class:`Classifier` or :class:`Regressor`)
