Global Variables
================

The sam_ml library has some global variables that one can set to modify output or change default values of functions.

.. note::

   You have to set the global variables before the first import of the **sam_ml** library if you want to use them.

Logging
-------

You can set the log level of the sam_ml library with the environment variable ``SAM_ML_LOG_LEVEL``. *(the default value is log level "info")*

Example:

.. code-block:: python

    import os
    os.environ["SAM_ML_LOG_LEVEL"] = "debug"

    from sam_ml.models.classifier import LR

n_jobs
------

Some functions in sklearn like ``crossvalidate`` and also some models like ``RandomForestClassifier`` have a parameter **n_jobs** for parallel running. In sam_ml, this is set to *"-1"* (use all capacity) as default. You can change it with the global variable ``SAM_ML_N_JOBS``. Possible values are *"-1"*, *"none"* (no parallelisation), or *"<positive integer>"* (for number of workers).

Example:

.. code-block:: python

    import os
    os.environ["SAM_ML_N_JOBS"] = "-1"

    from sam_ml.models.classifier import LR

.. note::

   This was added because of some issues with the **n_jobs** parameter on a windows virtual machine.

Sounds On
---------

The functions in the class ``CTest`` make a microwave-finish-sound when they trained the model so that the Data Scientist knows that they are finished if he/she/they are doing something else in the train time *(e.g. coffee-break)*. In sam_ml, this is set to "True" *(make sound)* as default. You can change it with the global variable ``SAM_ML_SOUND_ON``. Possible values are *"True"* and *"False"*.

Example:

.. code-block:: python

    import os
    os.environ["SAM_ML_SOUND_ON"] = "True"

    from sam_ml.models.classifier import LR

.. note::

   This was added because of some issues when the running machine has no audio output *(e.g., GitHub actions)*.

.. _global-variable-scoring-section:

Scoring Variables
-----------------

The following global variables are for setting the :doc:`classification scoring parameters <scoring>` at the start of your code so that you do not have to worry about setting the parameters in every function. The values can be the following:

- ``SAM_ML_AVG`` has to be "none", "micro", "macro", "binary", or "weighted"
- ``SAM_ML_POS_LABEL`` has to be "-1" or a string of an integer greater or equal 0
- ``SAM_ML_SCORING`` has to be "precision", "recall", "accuracy", "s_score", or "l_score"
- ``SAM_ML_SECONDARY_SCORING`` has to be "none", "precision", or "recall"
- ``SAM_ML_STRENGTH`` has to be a string of an integer greater 0

Example:

.. code-block:: python

    import os
    os.environ["SAM_ML_AVG"] = "macro"
    os.environ["SAM_ML_POS_LABEL"] = "-1"
    os.environ["SAM_ML_SCORING"] = "s_score"
    os.environ["SAM_ML_SECONDARY_SCORING"] = "none"
    os.environ["SAM_ML_STRENGTH"] = "3"

    from sam_ml.models.classifier import LR
