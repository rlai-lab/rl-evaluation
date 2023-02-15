import numpy as np
from RlEvaluation._utils.jit import try2jit

@try2jit
def mean(arr: np.ndarray) -> float:
    return float(np.mean(arr))
