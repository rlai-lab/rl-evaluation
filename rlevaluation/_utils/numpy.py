import numpy as np

from collections.abc import Sequence

def pad_stack(arrs: Sequence[np.ndarray], fill_value: float) -> np.ndarray:
    max_len = max(a.shape[0] for a in arrs)

    return np.stack([pad(a, max_len, fill_value) for a in arrs], axis=0)

def pad(arr: np.ndarray, length: int, fill_value: float) -> np.ndarray:
    return np.pad(arr, (0, length - arr.shape[0]), mode='constant', constant_values=fill_value)
