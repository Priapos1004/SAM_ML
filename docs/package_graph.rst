Package Graph
=============

.. uml::

    @startuml
    skinparam classAttributeIconSize 0
    hide empty members

    package "sam_ml" {

        package "sam_ml.data" {
            class Data
            package "sam_ml.data.preprocessing" {
                circle " "
            }
        }

        package "sam_ml.models" {
            class Model
            class Classifier
            class Regressor
            class AutoML
            package "sam_ml.models.classifier" {
                circle "  "
            }
            package "sam_ml.models.regressor" {
                circle "   "
            }
            package "sam_ml.models.automl" {
                circle "    "
        }
        }
    }

    AutoML <|-- "    "
    Data <|-- " "
    Model <|-- Classifier
    Model <|-- Regressor
    Classifier <|-- "  "
    Regressor <|-- "   "

    @enduml
