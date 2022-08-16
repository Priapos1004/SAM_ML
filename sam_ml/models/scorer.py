import math

from sklearn.metrics import precision_score, recall_score


def samuel_function(x: float) -> float:
    return math.sqrt(1/(1 + math.e**(12*(0.5-x))))

def lewis_function(x: float) -> float:
    return 1-(0.5-0.5*math.cos((x-1)*math.pi))**4


def s_scoring(y_true: list, y_pred: list, scoring: str = None, pos_label: int = -1, strength: int = 2, score_func = samuel_function) -> float:
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

        score_func: function to use for scoring (default: samuel_function)

    @return:
        score as float between 0 and 1
    """
    prec = precision_score(y_true, y_pred, average=None)
    rec = recall_score(y_true, y_pred, average=None)

    score = 1.0
    for i in range(len(prec)):
        if (scoring=='precision' and pos_label==i) or (scoring=='precision' and type(pos_label)==int and pos_label<=0) or (scoring==None and pos_label==i):
            score *= score_func(prec[i])**strength
        else:
            score *= score_func(prec[i])
    for i in range(len(rec)):
        if (scoring=='recall' and pos_label==i) or (scoring=='recall' and type(pos_label)==int and pos_label<=0) or (scoring==None and pos_label==i):
            score *= score_func(rec[i])**strength
        else:
            score *= score_func(rec[i])

    return score

def l_scoring(y_true: list, y_pred: list, scoring: str = None, pos_label: int = -1, strength: int = 2, score_func = lewis_function) -> float:
    return s_scoring(y_true, y_pred, scoring, pos_label, strength, score_func)
