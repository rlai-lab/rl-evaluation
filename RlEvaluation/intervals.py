import numpy as np
from numba import prange
from RlEvaluation._utils.data import compileReducer, Reducer
from RlEvaluation._utils.jit import try2pjit

RandomState = np.random.RandomState

def bootstrap(data: np.ndarray, statistic: Reducer = np.mean, coverage: float = 0.95, bootstraps: int = 2000, rng: RandomState = RandomState()):
    assert data.ndim == 2

    return _bootstrap(
        data=data,
        statistic=compileReducer(statistic),
        coverage=coverage,
        bootstraps=bootstraps,
        seed=rng.randint(2**32),
    )

@try2pjit
def _bootstrap(data: np.ndarray, statistic: Reducer, coverage: float, bootstraps: int, seed: int):
    np.random.seed(seed)
    samples, measurements = data.shape

    alpha = (1 - coverage) / 2

    out = np.empty((3, measurements))
    for i in prange(measurements):
        bs = np.empty(bootstraps)
        for j in range(bootstraps):
            sub = np.random.choice(data[:, i], size=samples, replace=True)
            bs[j] = statistic(sub)

        out[0, i] = np.percentile(bs, 100 * alpha)
        out[1, i] = statistic(data[:, i])
        out[2, i] = np.percentile(bs, 100 * (1 - alpha))

    return out
