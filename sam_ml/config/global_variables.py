import os


def get_sound_on() -> bool:
    sound_on = os.getenv("SAM_ML_SOUND_ON")
    if str(sound_on).lower() == "true" or sound_on is None:
        return True
    elif str(sound_on).lower() == "false":
        return False
    else:
        raise ValueError(f"SAM_ML_SOUND_ON cannot be '{sound_on}' -> has to be 'True' or 'False'")

def get_n_jobs() -> int|None:
    n_jobs = os.getenv("SAM_ML_N_JOBS")
    if str(n_jobs) == "-1" or n_jobs is None:
        return -1
    elif str(n_jobs).lower() == "none":
        return None
    elif str(n_jobs).isdigit():
        return int(n_jobs)
    else:
        raise ValueError(f"SAM_ML_N_JOBS cannot be '{n_jobs}' -> has to be 'none', '*positive integer*', or '-1'")
    