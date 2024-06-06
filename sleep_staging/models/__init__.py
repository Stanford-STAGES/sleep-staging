from .stages import StagesModel
from .usleep import USleepModel
from .utime import UTimeModel

available_models = {
    "stages": StagesModel,
    "usleep": USleepModel,
    "utime": UTimeModel,
}

# __all__ = ["available_models"]
