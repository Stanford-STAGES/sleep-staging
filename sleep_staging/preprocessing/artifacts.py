from typing import Dict, Union, List

import numpy as np


def artifact_detection(eeg: np.array, eog: np.array, emg: np.array) -> Dict[str, Union[List[float], float]]:
    """
    Function assumes input to be resampled, but not filtered.

    Args:
        eeg (np.array): 2xN
        eog (np.array): 2xN
        emg (np.array): 1xN

    Returns:
        Dict[str, Union[List[float], float]]: {eeg: 2x1, eog: 2x1, emg: 1}
    """
    pass
