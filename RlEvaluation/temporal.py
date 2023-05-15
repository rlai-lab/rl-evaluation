# TODO: steady-state performance
#  - Fit a model (piecewise linear with one node?) and report bias unit for second part

import numpy as np
import pandas as pd

from typing import Callable

TimeSummary = Callable[[np.ndarray | pd.Series], np.ndarray]

def mean(data: np.ndarray | pd.Series):
    assert len(data.shape) == 2
    return np.mean(data, axis=1)
