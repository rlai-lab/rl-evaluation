import numpy as np
import numba

from functools import partial
from typing import Any

njit = partial(numba.njit, fastmath=True, nogil=True, cache=True)

@njit(cache=False)
def over_first_axis(data: np.ndarray, f: Any):
    out = np.empty(data.shape[0])
    for i in range(data.shape[0]):
        out[i] = f(data[i])

    return out
