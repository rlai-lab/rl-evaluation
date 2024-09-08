import numpy as np
from typing import Callable, Sequence, Union
from rlevaluation._utils.jit import try2jit_no_cache

Numeric = Union[np.ndarray, float]
Reducer = Callable[[np.ndarray], Numeric]

def compileReducer(f: Reducer):
    # normalize return type of the reducer
    return try2jit_no_cache(lambda arr: float(f(arr)))

def ragged_to_square(seq: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    if np.isscalar(seq[0]):
        return np.expand_dims(seq, axis=1)

    m = max(len(x) for x in seq)
    out = np.empty((len(seq), m), dtype=seq[0].dtype)

    for i, x in enumerate(seq):
        assert x.ndim == 1

        needed = m - len(x)
        if needed > 0:
            out[i, :-needed] = x
            out[i, -needed:] = np.nan
        else:
            out[i] = x

    return out
