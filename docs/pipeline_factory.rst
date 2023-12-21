Pipeline Factory
================

The ``create_pipeline`` function dynamically creates a machine learning pipeline based on the input model.
All functions of the model (also special ones like ``plot_tree`` from :class:`DTC`) can be used with the pipeline. You can use the Pipeline for both :doc:`Classifier <classifier/index>` and :doc:`Regressor <regressor/index>`.

.. autofunction:: sam_ml.models.create_pipeline
