import numpy as np

from numba import prange
from typing import Any, NamedTuple

import rlevaluation.backend.statistics as bs
import rlevaluation._utils.numba as nbu


# ------------------------
# -- Temporal Summaries --
# ------------------------

def mean(data: np.ndarray):
    assert len(data.shape) == 2
    return np.nanmean(data, axis=1)

def agg(data: np.ndarray):
    assert len(data.shape) == 2
    return np.nansum(data, axis=1)


# -----------------------------
# -- Statistical Simulations --
# -----------------------------

@nbu.njit(parallel=True)
def curve_percentile_bootstrap_ci(
    rng: np.random.Generator,
    y: np.ndarray,
    statistic: Any,
    alpha: float = 0.05,
    iterations: int = 10000,
):
    n_measurements = y.shape[1]

    lo = np.empty(n_measurements, dtype=np.float64)
    center = np.empty(n_measurements, dtype=np.float64)
    hi = np.empty(n_measurements, dtype=np.float64)

    for i in prange(n_measurements):
        res = bs.percentile_bootstrap_ci(
            rng,
            y[:, i],
            statistic=statistic,
            alpha=alpha,
            iterations=iterations,
        )

        lo[i] = res.ci[0]
        center[i] = res.sample_stat
        hi[i] = res.ci[1]

    return CurvePercentileBootstrapResult(
        sample_stat=center,
        ci=(lo, hi),
    )


class CurvePercentileBootstrapResult(NamedTuple):
    sample_stat: np.ndarray
    ci: tuple[np.ndarray, np.ndarray]
