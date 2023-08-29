import numpy as np

import RlEvaluation._utils.numba as nbu

@nbu.njit(inline='always')
def sample_index(rng: np.random.Generator, probs: np.ndarray) -> int:
    p = rng.random()

    s = 0
    for i in range(len(probs)):
        s += probs[i]
        if s >= p:
            return i

    return len(probs) - 1
