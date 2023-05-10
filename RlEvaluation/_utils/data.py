import numpy as np
import pandas as pd
from typing import Callable, Sequence, Union
from RlEvaluation._utils.jit import try2jit_no_cache

Data = Union[np.ndarray, pd.DataFrame, pd.Series]
Numeric = Union[np.ndarray, float]
Reducer = Callable[[np.ndarray], Numeric]

def normalizeDataType(data: Data, dims: int, col: str = '') -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        assert col != ''
        d = data[col].to_numpy()
        data = ragged_to_square(d)

    elif isinstance(data, pd.Series):
        assert col != ''
        data = np.asarray(data[col])

    if np.ndim(data) > dims:
        raise Exception('Data has too many dimensions')

    elif np.ndim(data) < dims:
        needed = dims - np.ndim(data)
        shape = data.shape + tuple(1 for _ in range(needed))
        data = data.reshape(shape)

    assert np.ndim(data) == dims
    assert isinstance(data, np.ndarray)
    return data

def compileReducer(f: Reducer):
    # normalize return type of the reducer
    return try2jit_no_cache(lambda arr: float(f(arr)))

def ragged_to_square(l: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    m = max(len(x) for x in l)
    out = np.empty((len(l), m), dtype=l[0].dtype)

    for i, x in enumerate(l):
        assert x.ndim == 1

        needed = m - len(x)
        if needed > 0:
            out[i, :-needed] = x
            out[i, -needed:] = np.nan
        else:
            out[i] = x

    return out
