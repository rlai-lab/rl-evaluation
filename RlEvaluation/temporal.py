# TODO: steady-state performance
#  - Fit a model (piecewise linear with one node?) and report bias unit for second part

import enum
import numpy as np
import pandas as pd

from numba import prange
from typing import Any, List, Tuple, NamedTuple
from RlEvaluation.config import DataDefinition, maybe_global
from RlEvaluation.interpolation import Interpolation
from RlEvaluation.utils.pandas import subset_df

import RlEvaluation.statistics as Statistics
import RlEvaluation._utils.numba as nbu

def mean(data: np.ndarray):
    assert len(data.shape) == 2
    return np.nanmean(data, axis=1)


class TimeSummary(enum.Enum):
    mean = enum.member(mean)


def extract_learning_curves(
    df: pd.DataFrame,
    hyper_vals: Tuple[Any, ...],
    metric: str,
    data_definition: DataDefinition | None = None,
    interpolation: Interpolation | None = None,
):
    dd = maybe_global(data_definition)
    sub = subset_df(df, dd.hyper_cols, hyper_vals)

    groups = sub.groupby(dd.seed_col)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for _, group in groups:
        x = group[dd.time_col].to_numpy()
        y = group[metric].to_numpy()

        if interpolation is not None:
            x, y = interpolation(x, y)

        xs.append(x)
        ys.append(y)

    return xs, ys

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
        res = Statistics.percentile_bootstrap_ci(
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
    ci: Tuple[np.ndarray, np.ndarray]
