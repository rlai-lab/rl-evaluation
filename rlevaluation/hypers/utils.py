import numpy as np
import polars as pl
import rlevaluation._utils.numba as nbu
import rlevaluation._utils.numpy as npu

from typing import Any, Dict, Tuple, NamedTuple
from rlevaluation.config import DataDefinition, maybe_global


def group_measurements(df: pl.DataFrame, metric: str, data_definition: DataDefinition | None = None):
    dd = maybe_global(data_definition)

    idx2hypers: Dict[int, Any] = {}
    idx2measurements: Dict[int, np.ndarray] = {}

    cols = set(dd.hyper_cols).intersection(df.columns)

    for i, (n, group) in enumerate(df.group_by(list(cols))):
        idx2hypers[i] = n

        seed_data = []
        for _, run_data in group.group_by(dd.seed_col):
            seed_data.append(run_data[metric].to_numpy())

        idx2measurements[i] = npu.pad_stack(seed_data, fill_value=np.nan)

    return MeasurementGroup(
        idx2hypers=idx2hypers,
        idx2measurements=idx2measurements,
        n=len(idx2hypers),
    )

class MeasurementGroup(NamedTuple):
    idx2hypers: Dict[int, Tuple[Any, ...]]
    idx2measurements: Dict[int, np.ndarray]
    n: int


@nbu.njit(inline='always')
def idx_preference(prefer: str, a: np.ndarray):
    if prefer == 'high':
        return np.argmax(a)

    return np.argmin(a)
