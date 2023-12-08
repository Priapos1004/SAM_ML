Welcome to sam-ml-py's documentation!
=====================================

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: 💡 General Information
   
   swig_installation
   scoring
   global_variables

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🤖 Models

   classifier/index
   regressor/index
   auto_ml/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 📊 Data

   preprocessing/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: 🧩 Abstract classes
   
   abstract_classes/Model_class
   abstract_classes/Regressor_class
   abstract_classes/Classifier_class
   abstract_classes/Data_class
   abstract_classes/AutoML_class

|PyPI version| |docs| |CodeQuality| |view GitHub|

sam_ml is a machine-learning library created as an API that shall help to make modelling easier. 
It has different preprocessing steps, classifiers, regressors, and auto-ml approaches implemented.

Installation
------------

There are two versions of sam_ml - with and without the `SMAC <https://github.com/automl/SMAC3>`__ library for hyperparameter-tuning.

with SMAC
"""""""""

If you want to install SMAC, you first have to install ``swig`` (see :doc:`swig_installation`).

::

   pip install "sam-ml-py[with_swig]"

without SMAC
""""""""""""

::

   pip install sam-ml-py

Let's get started
-----------------

I recommend to read the advanced :doc:`scoring` documentation for classification problems because it is one of the major advantages of this library.

Here an example for binary classification with precision optimisation of label 1 (underrepresented):

>>> # load data (replace with own data)
>>> import pandas as pd
>>> from sklearn.datasets import make_classification
>>> from sklearn.model_selection import train_test_split
>>> X, y = make_classification(n_samples=3000, n_features=4, n_classes=2, weights=[0.9], random_state=42)
>>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(y)
>>> x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=42)
>>> 
>>> # start modelling
>>> from sam_ml.models.classifier import CTest
>>> 
>>> # initialise auot-ml class
>>> ctest = CTest(models = "all", scaler = "standard", selector = None, sampler = "ros")
>>> 
>>> # start randomCVsearch with 5 configurations per model type and evaluate the best parameters
>>> ctest.find_best_model_randomCV(x_train,y_train,x_test,y_test, scoring="s_score", avg="binary", pos_label=1, secondary_scoring="precision", strength=3, n_trails=5, cv_num=3)
>>> 
>>> # output and sort results
>>> score_df = ctest.output_scores_as_pd(sort_by=["s_score", "train_time"])
randomCVsearch (LogisticRegression (vec=None, scaler=standard, selector=None, sampler=ros)): 100%|██████████| 5/5 [00:00<00:00, 13.74it/s]
2023-12-08 21:12:57,721 - sam_ml.models.main_auto_ml - INFO - LogisticRegression (vec=None, scaler=standard, selector=None, sampler=ros) - score: 0.8114282933429915 (s_score) - parameters: {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}
<BLANKLINE>
randomCVsearch (QuadraticDiscriminantAnalysis (vec=None, scaler=standard, selector=None, sampler=ros)): 100%|██████████| 5/5 [00:00<00:00, 19.47it/s]
2023-12-08 21:12:58,010 - sam_ml.models.main_auto_ml - INFO - QuadraticDiscriminantAnalysis (vec=None, scaler=standard, selector=None, sampler=ros) - score: 0.8788135203591323 (s_score) - parameters: {'reg_param': 0.0}
<BLANKLINE>
...
<BLANKLINE>                  
                                                      accuracy    precision   recall      s_score     l_score     train_time  train_score  best_score (rCVs)  best_hyperparameters (rCVs)
AdaBoostClassifier (DTC based) (vec=None, scale...    0.983333    0.943396    0.877193    0.984656    0.999998    0:00:02     0.995061     0.985320           {'algorithm': 'SAMME', 'estimator__max_depth':...
AdaBoostClassifier (RFC based) (vec=None, scale...    0.983333    0.943396    0.877193    0.984656    0.999998    0:00:01     0.995061     0.984980           {'algorithm': 'SAMME', 'estimator__max_depth':...
XGBClassifier (vec=None, scaler=standard, selec...    0.981667    0.942308    0.859649    0.983298    0.999995    0:00:00     0.994929     0.985982           {'colsample_bytree': 1.0, 'gamma': 0.0, 'learn...
KNeighborsClassifier (vec=None, scaler=standard...    0.980000    0.909091    0.877193    0.980948    0.999998    0:00:00     0.995061     0.978702           {'leaf_size': 37, 'n_neighbors': 2, 'p': 1, 'w...
...

.. |PyPI version| image:: https://badge.fury.io/py/sam-ml-py.svg
   :target: https://badge.fury.io/py/sam-ml-py
.. |docs| image:: https://github.com/priapos1004/SAM_ML/workflows/docs/badge.svg
   :target: https://priapos1004.github.io/SAM_ML/
.. |CodeQuality| image:: https://github.com/priapos1004/SAM_ML/workflows/Code%20Quality%20Checks/badge.svg
   :target: https://github.com/Priapos1004/SAM_ML/actions/workflows/CodeQualityChecks.yml
.. |view GitHub| image:: https://img.shields.io/badge/View_on-GitHub-black?style=flat-square&logo=github
   :target: https://github.com/Priapos1004/SAM_ML
