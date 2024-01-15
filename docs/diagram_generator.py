import os
from pathlib import Path

from plantuml import PlantUML

# PlantUML server URL or local JAR file path
plantuml_server_url = 'http://www.plantuml.com/plantuml/svg/'

# Initialize PlantUML object with server URL
plantuml = PlantUML(url=plantuml_server_url)


def generate_uml_diagram():
    # PlantUML diagram code
    uml_code = """
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
                class ClassifierPipeline
                class RegressorPipeline
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
        Classifier <|-- ClassifierPipeline
        Regressor <|-- RegressorPipeline
        Regressor <|-- "   "
        ClassifierPipeline <-- create_pipeline
        RegressorPipeline <-- create_pipeline
        BasePipeline <|-- ClassifierPipeline
        BasePipeline <|-- RegressorPipeline

    @enduml
    """

    # If the folder does not exist, create it
    if not Path("docs/_images/").exists():
        Path("docs/_images/").mkdir(parents=True)

    # Generate the diagram and save it as an SVG file
    svg_output = plantuml.processes(uml_code)
    if svg_output:
        Path("docs/_images/uml_diagram.svg").write_bytes(svg_output)
        print("SVG diagram created successfully.")
    else:
        print("Failed to create SVG diagram.")


if __name__ == "__main__":
    generate_uml_diagram()
