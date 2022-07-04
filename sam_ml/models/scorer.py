import math

from sklearn.metrics import precision_score, recall_score


def samuel_function(x: float) -> float:
    return math.sqrt(1/(1 + math.e**(12*(0.5-x))))


def s_scoring(y_true: list, y_pred: list, scoring: str = None, pos_label: int = -1, strength: int = 2) -> float:
    """
    @param:
        y_true, y_pred: data to evaluate on
        
        scoring:
            None: no preference between precision and recall
            'precision': take precision more into account
            'recall': take recall more into account
        
        pos_label:
            pos_label > 0: take <scoring> in class <pos_label> more into account
            pos_label = -1: handle all classes the same

        strength: higher strength means a higher weight for the prefered scoring/pos_label

    @return:
        score as float between 0 and 1
    """
    prec = precision_score(y_true, y_pred, average=None)
    rec = recall_score(y_true, y_pred, average=None)

    score = 1.0
    for i in range(len(prec)):
        if (scoring=='precision' and pos_label==i) or (scoring=='precision' and pos_label<=0) or (scoring==None and pos_label==i):
            score *= samuel_function(prec[i])**strength
        else:
            score *= samuel_function(prec[i])
    for i in range(len(rec)):
        if (scoring=='recall' and pos_label==i) or (scoring=='recall' and pos_label<=0) or (scoring==None and pos_label==i):
            score *= samuel_function(rec[i])**strength
        else:
            score *= samuel_function(rec[i])

    return score


def lewis_function(x: float) -> float:
    return 1-(0.5-0.5*math.cos((x-1)*math.pi))**4

def comb_scoring(prec: list[float], rec: list[float], func) -> float:
    total = 1.0
    for i in prec:
        total *= func(i)
    for i in rec:
        total *= func(i)
    return total

def l_scoring(y_true: list, y_pred: list) -> float:
        prec = precision_score(y_true, y_pred, average=None)
        rec = recall_score(y_true, y_pred, average=None)
        cs = comb_scoring(prec, rec, lewis_function)
        return cs
