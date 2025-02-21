import numpy as np
import polars as pl

from numba.typed import List as NList
from typing import Any, NamedTuple

from rlevaluation.config import DataDefinition, maybe_global
from rlevaluation.statistics import Statistic
from rlevaluation.temporal import TimeSummary

from rlevaluation.hypers.format import make_score_format
from rlevaluation.hypers.interface import Preference
from rlevaluation.hypers.simulation import bootstrap_hyper_selection

def select_best_hypers(
    df: pl.DataFrame,
    metric: str,
    prefer: Preference,
    threshold: float = 0.05,
    statistic: Statistic = Statistic.mean,
    time_summary: TimeSummary = TimeSummary.mean,
    data_definition: DataDefinition | None = None,
):
    """
    Produces a report on the best performing hyperparameters given a dataframe
    with the following format:
      stepsize  seed  metric  time
      0.01      0     0.1     1
      0.01      0     0.2     2
      ...

    with a column for every hyperparameter.

    Uses a bootstrap procedure to estimate the uncertainty in the
    hyperparameter selection process, If there are insufficient samples
    to confidentally select the best performing hyperparameter configuration
    from the given data, then multiple hyperparameter configurations will
    be returned alongside the corresponding probability that each of these
    is the best configuration.

    Note that it is statistically invalid to analyze only the single
    best performing hyperparameter when there is uncertainty in the
    selection process.
    """
    dd = maybe_global(data_definition)

    cols = set(dd.hyper_cols).intersection(df.columns)

    df = (
        make_score_format(df, metric, dd, time_summary)
        .group_by(cols)
        .agg(pl.col(metric))
    )

    score_per_seed = df[metric].to_numpy()
    score_per_seed = NList(score_per_seed)

    rng = np.random.default_rng(0)
    out = bootstrap_hyper_selection(rng, score_per_seed, statistic.value, prefer.value, threshold)
    config = {
        col: df[col][out.best_idx] for col in cols
    }

    return HyperSelectionResult(
        best_configuration=config,
        best_score=out.best_score,

        uncertainty_set_configurations=[
            {col: df[col][int(idx)] for col in cols}
            for idx in out.uncertainty_set_idxs
        ],
        uncertainty_set_probs=out.uncertainty_set_probs,
        sample_stat=out.sample_stat,
        ci=out.ci,
        config_params=list(cols),
    )

class HyperSelectionResult(NamedTuple):
    best_configuration: dict[str, Any]
    best_score: float

    uncertainty_set_configurations: list[dict[str, Any]]
    uncertainty_set_probs: np.ndarray
    sample_stat: float
    ci: tuple[float, float]
    config_params: list[str]
