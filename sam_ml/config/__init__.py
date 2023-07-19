from .global_variables import (
    get_avg,
    get_n_jobs,
    get_pos_label,
    get_scoring,
    get_secondary_scoring,
    get_sound_on,
    get_strength,
)
from .logging import setup_logger

__all__ = {
    "logger setup function": "setup_logger",
    "get n_jobs parameter": "get_n_jobs",
    "get if sound on": "get_sound_on",
    "get avg parameter": "get_avg",
    "get pos_label parameter": "get_pos_label",
    "get scoring parameter": "get_scoring",
    "get secondary_scoring parameter": "get_secondary_scoring",
    "get strength parameter": "get_strength",
}
