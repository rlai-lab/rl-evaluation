import numpy as np
import pandas as pd
from typing import Callable, Sequence, Set, Union
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
    if np.isscalar(l[0]):
        return np.expand_dims(l, axis=1)

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

def get_other_cols(df: pd.DataFrame, known: Set[str], seed: str):
    cols = set(df.columns)
    return cols - (known | {seed})

def is_wide_format(df: pd.DataFrame, metrics: Set[str], seed_col: str):
    if seed_col in df:
        return False

    if any(not isinstance(df[m].iloc[0], np.ndarray) for m in metrics):
        return False

    return True

def make_wide_format(df: pd.DataFrame, hypers: Set[str], metrics: Set[str], seed_col: str):
    if is_wide_format(df, metrics, seed_col):
        return df

    op = {
        c: lambda x: np.array(list(x))
        for c in metrics
    }

    df = df.groupby(list(hypers)).agg(op).reset_index()
    return df
