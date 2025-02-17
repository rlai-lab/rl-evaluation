import enum
import numpy as np
import rlevaluation.backend.statistics as bs

from typing import Any


# ----------------------
# -- Basic Statistics --
# ----------------------

class Statistic(enum.Enum):
    mean = enum.member(bs.mean)
    sum = enum.member(bs.agg)


# -----------------------------
# -- Statistical Simulations --
# -----------------------------

def percentile_bootstrap_ci(
    rng: np.random.Generator,
    a: np.ndarray,
    statistic: Statistic = Statistic.mean,
    alpha: float = 0.05,
    iterations: int = 10000,
):
    f: Any = statistic.value
    return bs.percentile_bootstrap_ci(
        rng=rng,
        a=a,
        statistic=f,
        alpha=alpha,
        iterations=iterations,
    )


def stratified_percentile_bootstrap_ci(
    rng: np.random.Generator,
    a: np.ndarray | list[np.ndarray],
    class_probs: np.ndarray,
    statistic: Statistic = Statistic.mean,
    alpha: float = 0.05,
    iterations: int = 10000,
):
    f: Any = statistic.value
    return bs.stratified_percentile_bootstrap_ci(
        rng=rng,
        a=a,
        class_probs=class_probs,
        statistic=f,
        alpha=alpha,
        iterations=iterations,
    )
