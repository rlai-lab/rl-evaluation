# TODO: steady-state performance
#  - Fit a model (piecewise linear with one node?) and report bias unit for second part

import enum
import numpy as np
import pandas as pd

from typing import Any, List, Sequence, Tuple
from RlEvaluation.config import DataDefinition, maybe_global
from RlEvaluation.interpolation import Interpolation
from RlEvaluation.statistics import Statistic
from RlEvaluation.utils.pandas import subset_df

import RlEvaluation.backend.temporal as bt

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
    df: pd.DataFrame,
    hyper_vals: Tuple[Any, ...],
    metric: str,
    data_definition: DataDefinition | None = None,
    interpolation: Interpolation | None = None,
):
    dd = maybe_global(data_definition)
    cols = set(dd.hyper_cols).intersection(df.columns)
    sub = subset_df(df, list(cols), hyper_vals)

    groups = sub.groupby(dd.seed_col)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for _, group in groups:
        non_na = group[group[metric].notna()]
        x = non_na[dd.time_col].to_numpy().astype(np.int64)
        y = non_na[metric].to_numpy().astype(np.float64)

        if interpolation is not None:
            x, y = interpolation(x, y)

        xs.append(x)
        ys.append(y)

    return xs, ys

def extract_multiple_learning_curves(
    df: pd.DataFrame,
    hyper_vals: Sequence[Tuple[Any, ...]],
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
