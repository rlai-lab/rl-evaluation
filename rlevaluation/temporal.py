# TODO: steady-state performance
#  - Fit a model (piecewise linear with one node?) and report bias unit for second part

from collections.abc import Sequence
import enum
import numpy as np
import polars as pl
import rlevaluation._utils.dict as du

from typing import Any
from rlevaluation.config import DataDefinition, maybe_global
from rlevaluation.interpolation import Interpolation
from rlevaluation.statistics import Statistic

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
    hyper_vals: dict[str, Any],
    metric: str,
    data_definition: DataDefinition | None = None,
    interpolation: Interpolation | None = None,
):
    """
    Takes a dataframe of shape:
      stepsize  seed  metric  time
      0.01      0     0.1     1
      0.01      0     0.2     2
      0.01      0     0.3     3

    and extracts a learning curve per seed for the given hyperparameters,
    where a "learning curve" is defined by a numpy array for x
    values and another y values.
    """
    dd = maybe_global(data_definition)
    cols = set(dd.hyper_cols).intersection(df.columns)

    sub = df.filter(**du.subset(hyper_vals, cols))
    groups = sub.group_by(dd.seed_col)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
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
    hyper_vals: Sequence[dict[str, Any]],
    metric: str,
    data_definition: DataDefinition | None = None,
    interpolation: Interpolation | None = None,
):
    """
    Extracts a learning curve for multiple hyperparameter configurations
    from a dataframe of shape
      stepsize  seed  metric  time
      0.01      0     0.1     1
      0.01      0     0.2     2
      0.01      0     0.3     3

    returning two lists of numpy arrays, one for x and one for y.
    Each list of of length num_seeds * num_hypers (assuming same
    number of seeds per hyperparameter configuration).
    """
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
    """
    Produces the (1-alpha) confidence interval for each point in a learning
    curve using the percentile bootstrap method --- a non-parametric CI that
    does not require any assumptions on the underlying data.
    """
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
    """
    Produces the (1-alpha) confidence level beta-tolerance interval for each point
    across a set of learning curves. The tolerance interval states that with (1-alpha)
    confidence, this interval captures beta proportion of the data.

    By default, this corresponds to 90% data coverage with 95% confidence.
    """
    return bs.tolerance_interval_curve(
        y,
        alpha,
        beta,
    )
