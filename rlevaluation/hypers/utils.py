import numpy as np
import rlevaluation._utils.numba as nbu

@nbu.njit(inline='always')
def idx_preference(prefer: str, a: np.ndarray):
    if prefer == 'high':
        return np.argmax(a)

    return np.argmin(a)
