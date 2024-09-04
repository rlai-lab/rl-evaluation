import numpy as np
import pandas as pd

from typing import Callable, Tuple

import rlevaluation._utils.numba as nbu


Interpolation = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


@nbu.njit
def compute_step_return(t: np.ndarray, m: np.ndarray, max_length: int) -> Tuple[np.ndarray, np.ndarray]:
    assert t.shape == m.shape
    assert np.all(t[1:] > t[:-1]), 'Time column should be strictly increasing'

    x = np.arange(max_length)
    out = np.empty(max_length, dtype=m.dtype)

    last = 0
    for i in range(t.shape[0]):
        out[last:t[i]] = m[i]
        last = t[i]

    out[t[-1]:] = m[-1]

    return x, out

def compute_step_return_df(df: pd.DataFrame, time: str, metric: str, max_length: int):
    t = df[time].to_numpy()
    m = df[metric].to_numpy()

    return compute_step_return(t, m, max_length)
