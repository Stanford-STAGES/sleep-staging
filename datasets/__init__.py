from .isrc import ISRCDataset
from .khc import KoreanDataset
from .ssc_wsc_psg import SscWscDataModule
from .stages import STAGESDataModule

available_datamodules = {
    "stages": STAGESDataModule,
    "ssc-wsc": SscWscDataModule,
}
# __all__ = ["ISRCDataset", "KoreanDataset", "SscWscPsgDataset"]
