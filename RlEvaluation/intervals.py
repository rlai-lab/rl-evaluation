import numpy as np
from numba import prange
from RlEvaluation._utils.data import normalizeDataType, compileReducer, Data, Reducer
from RlEvaluation._utils.jit import try2pjit

RandomState = np.random.RandomState

def bootstrap(data: Data, column: str = '', statistic: Reducer = np.mean, coverage: float = 0.95, bootstraps: int = 10000, rng: RandomState = RandomState()):
    data = normalizeDataType(data, dims=2, col=column)

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
