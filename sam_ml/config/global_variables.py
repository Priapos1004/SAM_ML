import os


def get_sound_on() -> bool:
    sound_on = os.getenv("SAM_ML_SOUND_ON")
    if str(sound_on).lower() == "true" or sound_on is None:
        return True
    elif str(sound_on).lower() == "false":
        return False
    else:
        raise ValueError(f"SAM_ML_SOUND_ON cannot be '{sound_on}' -> has to be 'True' or 'False'")

def get_n_jobs() -> int | None:
    n_jobs = os.getenv("SAM_ML_N_JOBS")
    if str(n_jobs) == "-1" or n_jobs is None:
        return -1
    elif str(n_jobs).lower() == "none":
        return None
    elif str(n_jobs).isdigit():
        return int(n_jobs)
    else:
        raise ValueError(f"SAM_ML_N_JOBS cannot be '{n_jobs}' -> has to be 'none', '*positive integer*', or '-1'")
    
def get_avg() -> str:
    avg = os.getenv("SAM_ML_AVG")
    if  avg is None:
        return "macro"
    elif str(avg) == "none":
        return None
    elif str(avg).lower() in ("micro", "macro", "binary", "weighted"):
        return str(avg)
    else:
        raise ValueError(f"SAM_ML_AVG cannot be '{avg}' -> has to be 'none', 'micro', 'macro', 'binary', or 'weighted'")
    
def get_pos_label() -> int:
    pos_label = os.getenv("SAM_ML_POS_LABEL")
    if str(pos_label) == "-1" or pos_label is None:
        return -1
    elif str(pos_label).isnumeric():
        return int(pos_label)
    else:
        raise ValueError(f"SAM_ML_POS_LABEL cannot be '{pos_label}' -> has to be '-1' or an integer greater or equal 0")
    
def get_scoring() -> str:
    scoring = os.getenv("SAM_ML_SCORING")
    if scoring is None:
        return "accuracy"
    elif str(scoring) in ("precision", "recall", "accuracy", "s_score", "l_score"):
        return str(scoring)
    else:
        raise ValueError(f"SAM_ML_SCORING cannot be '{scoring}' -> has to be 'precision', 'recall', 'accuracy', 's_score', or 'l_score'")
    
def get_secondary_scoring() -> str | None:
    secondary_scoring = os.getenv("SAM_ML_SECONDARY_SCORING")
    if secondary_scoring is None or str(secondary_scoring) == "none":
        return None
    elif str(secondary_scoring) in ("precision", "recall"):
        return str(secondary_scoring)
    else:
        raise ValueError(f"SAM_ML_SECONDARY_SCORING cannot be '{secondary_scoring}' -> has to be 'none', 'precision' or 'recall'")
    
def get_strength() -> int:
    strength = os.getenv("SAM_ML_STRENGTH")
    if strength is None:
        return 3
    elif str(strength).isnumeric():
        return int(strength)
    else:
        raise ValueError(f"SAM_ML_STRENGTH cannot be '{strength}' -> has to be positive integer greater-equal 1")
    