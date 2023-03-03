import numpy as np
from typing import Any
from RlEvaluation._utils.jit import try2jit


@try2jit
def over_first_axis(data: np.ndarray, f: Any):
    out = np.empty(data.shape[0])
    for i in range(data.shape[0]):
        out[i] = f(data[i])

    return out
