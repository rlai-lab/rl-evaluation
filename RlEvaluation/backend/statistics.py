import numpy as np
import RlEvaluation._utils.numba as nbu

from typing import Any, Callable, List, NamedTuple, Tuple


# ----------------------
# -- Basic Statistics --
# ----------------------

@nbu.njit(inline='always')
def mean(a: np.ndarray, axis: int = 0):
    return np.sum(a, axis=axis) / a.shape[axis]

@nbu.njit(inline='always')
def agg(a: np.ndarray, axis: int = 0):
    return np.sum(a, axis=axis)


# -----------------------------
# -- Statistical Simulations --
# -----------------------------

@nbu.njit
def percentile_bootstrap_ci(
    rng: np.random.Generator,
    a: np.ndarray,
    statistic: Callable[[np.ndarray], Any] = mean,
    alpha: float = 0.05,
    iterations: int = 10000,
):
    bs = np.empty(iterations, dtype=np.float64)

    for i in range(iterations):
        idxs = rng.integers(0, len(a), size=len(a))
        bs[i] = statistic(a[idxs])

    sample_stat = statistic(a)

    lo_b = (alpha / 2)
    hi_b = 1 - (alpha / 2)
    lo, hi = np.percentile(bs, (100 * lo_b, 100 * hi_b))

    return PercentileBootstrapResult(
        sample_stat=sample_stat,
        ci=(lo, hi),
    )


@nbu.njit
def stratified_percentile_bootstrap_ci(
    rng: np.random.Generator,
    a: np.ndarray | List[np.ndarray],
    class_probs: np.ndarray,
    statistic: Callable[[np.ndarray], Any] = mean,
    alpha: float = 0.05,
    iterations: int = 10000,
):
    bs = np.empty(iterations, dtype=np.float64)

    samples = sum([len(sub) for sub in a])
    c_samples = [int(max(1, p * samples)) for p in class_probs]

    # this may not be exactly equal to `samples` due to the max(1, ...)
    sub_samples = sum(c_samples)

    for i in range(iterations):
        sub_data = np.empty(sub_samples, dtype=np.float64)
        acc = 0
        for c in range(len(class_probs)):
            idxs = rng.integers(0, len(a[c]), size=c_samples[c])

            sub_data[acc:acc + c_samples[c]] = a[c][idxs]
            acc += c_samples[c]

        bs[i] = statistic(sub_data)

    sample_stat = bs.mean()
    lo_b = 100 * (alpha / 2)
    hi_b = 100 - 100 * (alpha / 2)
    lo, hi = np.percentile(bs, (lo_b, hi_b))

    return PercentileBootstrapResult(
        sample_stat=sample_stat,
        ci=(lo, hi),
    )


class PercentileBootstrapResult(NamedTuple):
    sample_stat: float
    ci: Tuple[float, float]
