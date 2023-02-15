import numpy as np
import pandas as pd
from typing import Callable, Union
from RlEvaluation._utils.jit import try2jit_no_cache

Data = Union[np.ndarray, pd.DataFrame]
Numeric = Union[np.ndarray, float]
Reducer = Callable[[np.ndarray], Numeric]

def normalizeDataType(data: Data, dims: int, col: str = '') -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        assert col != ''
        data = data[col].to_numpy()

    elif isinstance(data, pd.Series):
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
