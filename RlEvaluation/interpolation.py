import numpy as np
import pandas as pd

from typing import Callable, Tuple

import RlEvaluation._utils.numba as nbu


Interpolation = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


@nbu.njit
def compute_step_return(t: np.ndarray, m: np.ndarray, max_length: int) -> Tuple[np.ndarray, np.ndarray]:
    assert t.shape == m.shape
    x = np.arange(max_length)
    out = np.empty(max_length, dtype=m.dtype)

    acc = 0
    for i in range(t.shape[0]):
        steps = t[i]
        n = acc + steps

        out[acc:n] = m[i]
        acc = n

    out[acc:] = m[-1]

    return x, out

def compute_step_return_df(df: pd.DataFrame, time: str, metric: str, max_length: int):
    t = df[time].to_numpy()
    m = df[metric].to_numpy()

    return compute_step_return(t, m, max_length)
