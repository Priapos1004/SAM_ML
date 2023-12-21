Scoring
=======

.. _classifier-scoring-section:

Classification
--------------

In this library, different metrics are used for evaluating models. The parameters used for this are the following:

- **scoring**: ``str`` - parameter to define scoring (e.g., which score to use to optimize hyperparameters)
- **avg**: ``str`` - how shall the average of precision and recall be calculated (e.g., "micro", "weighted", "binary")
- **pos_label**: ``int`` or ``str`` - which class to prioritize
- **secondary_scoring**: ``str`` - weights the scoring (only for scoring='s_score'/'l_score') *(see below)*
- **strength**: ``int`` - higher weight for the preferred secondary_scoring/pos_label (only for scoring='s_score'/'l_score') *(see below)*

The scores are **accuracy**, **precision**, **recall**, **s_score**, and **l_score**.

.. note:: 
   
   You can also just ignore **avg**/ **pos_label** / **secondary_scoring** / **strength** and just use **scoring** with *precision* / *recall* / *accuracy*.

.. note:: 
   
   If you do not want to set the scoring variables in every function, you can set them once at the beginning of your code with :ref:`Global Variables <global-variable-scoring-section>`.

Advanced Scoring with s_score or l_score
""""""""""""""""""""""""""""""""""""""""

In addition to the "normal" metrics, the library also uses two scores `s_score` and `l_score` which help to improve the search for good models in hyperparameter search and finding the best model.

**Core idea:**

- **secondary_scoring** and **pos_label** to prioritize metrics/classes in optimization
- **strength** to set how much the prioritization is compared to scores in all the other metrics/classes
- always includes all metrics/classes and punishes really bad scores in one metric/class

Both scores use a function that is applied to precision and recall (rescales values), and afterwards, the precision and recall of the different classes are multiplied (potentiated with **strength** parameter for class **pos_label** and metrics **secondary_scoring**).

.. math::

   s(x) = \sqrt{\frac{1}{{1 + \exp\left(12\cdot(0.5-x)\right)}}}

.. math::

   l(x) = 1 - \left(0.5 - 0.5\cdot\cos\left((x-1)\cdot\pi\right)\right)^4

**Possible values:**

- **secondary_scoring**:

  - ``None`` : no preference between precision and recall
  - ``'precision'`` : take precision more into account
  - ``'recall'`` : take recall more into account

- **pos_label**:

  - ``pos_label > 0`` : take <secondary_scoring> in class <pos_label> more into account
  - ``pos_label = -1`` : handle all classes the same

- **strength**: higher strength means a higher weight for the preferred **secondary_scoring**/**pos_label**

**Example:**

I want to optimize *precision* in class 2 (classes 0, 1, 2, 3) and also not lose too much *recall* in optimization (prevent 100% *precision* and 0.01% *recall*).

.. code-block:: python

    scoring = "s_score"
    pos_label = 2
    secondary_scoring = "precision"
    strength = 4

Now I can play around with **strength** to prioritize *precision* stronger or less. 

.. note::

   The **avg** parameter does not affect *s_score* and *l_score*, but it is relevant for the calculation of the precision and recall scores in the results (e.g., **avg="macro"** can lead to misinterpretation of the results if the models perform in some classes very well and the other classes not at all)
