# TODO: steady-state performance
#  - Fit a model (piecewise linear with one node?) and report bias unit for second part

import enum
import numpy as np
import pandas as pd

from typing import Any, List, Sequence, Tuple
from rlevaluation.config import DataDefinition, maybe_global
from rlevaluation.interpolation import Interpolation
from rlevaluation.statistics import Statistic
from rlevaluation.utils.pandas import subset_df

import rlevaluation.backend.statistics as bs
import rlevaluation.backend.temporal as bt

# ------------------------
# -- Temporal Summaries --
# ------------------------

class TimeSummary(enum.Enum):
    mean = enum.member(bt.mean)
    sum = enum.member(bt.agg)


# ---------------------
# -- Data Management --
# ---------------------

def extract_learning_curves(
    df: pl.DataFrame,
    hyper_vals: Dict[str, Any],
    metric: str,
    data_definition: DataDefinition | None = None,
    interpolation: Interpolation | None = None,
):
    dd = maybe_global(data_definition)
    cols = set(dd.hyper_cols).intersection(df.columns)

    sub = df.filter(**du.subset(hyper_vals, cols))
    groups = sub.group_by(dd.seed_col)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for _, group in groups:
        non_na = group.drop_nulls(subset=metric)
        x = non_na[dd.time_col].to_numpy().astype(np.int64)
        y = non_na[metric].to_numpy().astype(np.float64)

        idx = np.argwhere(x[1:] <= x[:-1])

        if idx:
            x = x[:idx[0][0]]
            y = y[:idx[0][0]]

        if interpolation is not None:
            x, y = interpolation(x, y)

        xs.append(x)
        ys.append(y)

    return xs, ys

def extract_multiple_learning_curves(
    df: pl.DataFrame,
    hyper_vals: Sequence[Dict[str, Any]],
    metric: str,
    data_definition: DataDefinition | None = None,
    interpolation: Interpolation | None = None,
):
    out_xs = []
    out_ys = []

    for hypers in hyper_vals:
        xs, ys = extract_learning_curves(
            df,
            hypers,
            metric,
            data_definition,
            interpolation,
        )

        out_xs += xs
        out_ys += ys

    return out_xs, out_ys

# -----------------------------
# -- Statistical Simulations --
# -----------------------------

def curve_percentile_bootstrap_ci(
    rng: np.random.Generator,
    y: np.ndarray,
    statistic: Statistic = Statistic.mean,
    alpha: float = 0.05,
    iterations: int = 10000,
):
    f: Any = statistic.value
    return bt.curve_percentile_bootstrap_ci(
        rng=rng,
        y=y,
        statistic=f,
        alpha=alpha,
        iterations=iterations,
    )


def curve_tolerance_interval(
    y: np.ndarray,
    alpha: float = 0.05,
    beta: float = 0.9,
):
    return bs.tolerance_interval_curve(
        y,
        alpha,
        beta,
    )
