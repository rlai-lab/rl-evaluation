import numpy as np
import pandas as pd

from numba.typed import List as NList
from typing import Any, List, Tuple, NamedTuple

from rlevaluation.config import DataDefinition, maybe_global
from rlevaluation.statistics import Statistic
from rlevaluation.temporal import TimeSummary

from rlevaluation.hypers.interface import Preference
from rlevaluation.hypers.simulation import bootstrap_hyper_selection
from rlevaluation.hypers.utils import group_measurements

def select_best_hypers(
    df: pd.DataFrame,
    metric: str,
    prefer: Preference,
    threshold: float = 0.05,
    statistic: Statistic = Statistic.mean,
    time_summary: TimeSummary = TimeSummary.mean,
    data_definition: DataDefinition | None = None,
):
    dd = maybe_global(data_definition)
    f: Any = time_summary.value

    groups = group_measurements(df, metric=metric, data_definition=dd)
    score_per_seed = NList()
    for i in range(groups.n):
        data = groups.idx2measurements[i]

        agg_time = f(data)
        score_per_seed.append(agg_time)

    rng = np.random.default_rng(0)
    out = bootstrap_hyper_selection(rng, score_per_seed, statistic.value, prefer.value, threshold)

    cols = set(dd.hyper_cols).intersection(df.columns)
    return HyperSelectionResult(
        best_configuration=groups.idx2hypers[out.best_idx],
        best_score=out.best_score,

        uncertainty_set_configurations=[
            groups.idx2hypers[idx] for idx in out.uncertainty_set_idxs
        ],
        uncertainty_set_probs=out.uncertainty_set_probs,
        sample_stat=out.sample_stat,
        ci=out.ci,
        config_params=list(cols),
    )

class HyperSelectionResult(NamedTuple):
    best_configuration: Tuple[Any, ...]
    best_score: float

    uncertainty_set_configurations: List[Tuple[Any, ...]]
    uncertainty_set_probs: np.ndarray
    sample_stat: float
    ci: Tuple[float, float]
    config_params: List[str]
