from .massc import MasscModel
from .massc_attention import MASSCAttention
from .massc_average import MASSCAverage
from .massc_avgloss import AvgLossMasscModel
from .massc_new import MasscV2Model
from .massc_simple import SimpleMasscModel
from .stages import StagesModel
from .utime import UTimeModel

available_models = {
    "massc": MasscModel,
    "stages": StagesModel,
    "utime": UTimeModel,
    "massc_attention": MASSCAttention,
    "massc_average": MASSCAverage,
}

# __all__ = ["available_models"]
