import math

from sklearn.metrics import precision_score, recall_score


def samuel_function(x: float) -> float:
    return math.sqrt(1/(1 + math.e**(12*(0.5-x))))

def lewis_function(x: float) -> float:
    return 1-(0.5-0.5*math.cos((x-1)*math.pi))**4

def comb_scoring(prec: list[float], rec: list[float], func) -> float:
    total = 1
    for i in prec:
        total *= func(i)
    for i in rec:
        total *= func(i)
    return total

def s_scoring(y_true: list, y_pred: list) -> float:
        prec = precision_score(y_true, y_pred, average=None)
        rec = recall_score(y_true, y_pred, average=None)
        cs = comb_scoring(prec, rec, samuel_function)
        return cs

def l_scoring(y_true: list, y_pred: list) -> float:
        prec = precision_score(y_true, y_pred, average=None)
        rec = recall_score(y_true, y_pred, average=None)
        cs = comb_scoring(prec, rec, lewis_function)
        return cs
