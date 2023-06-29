import pandas as pd

from sam_ml.models import (
    ABC,
    BC,
    BNB,
    DTC,
    ETC,
    GBM,
    GNB,
    GPC,
    KNC,
    LDA,
    LR,
    LSVC,
    MLPC,
    QDA,
    RFC,
    SVC,
    XGBC,
    Pipeline,
)


def test_models():
    models = [ABC(), BC(), BNB(), DTC(), ETC(), GNB(), GPC(), GBM(), KNC(), LDA(), LSVC(), LR(), Pipeline(), MLPC(), QDA(), RFC(), SVC(), XGBC()]
    x = pd.DataFrame({"col1": [1,2,3,2,1,2,3,1,2,1], "col2": [4,5,4,3,4,3,4,4,3,5], "col3": [4,5,4,3,4,3,4,4,3,5]})
    y = pd.Series([1,2,1,0,2,0,2,1,2,0])
    for model in models:
        model.train(x, y, console_out=False)
        model.evaluate(x, y, console_out=False)
