from .isrc import ISRCDataModule
from .khc import KHCDataModule
from .ssc_wsc_psg import SscWscDataModule
from .stages import STAGESDataModule

available_datamodules = {
    "stages": STAGESDataModule,
    "ssc-wsc": SscWscDataModule,
    "khc": KHCDataModule,
    "massc_average": SscWscDataModule,
    "isrc": ISRCDataModule,
}
# __all__ = ["ISRCDataset", "KoreanDataset", "SscWscPsgDataset"]
