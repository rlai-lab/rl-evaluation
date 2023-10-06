import dataclasses

from typing import Iterable, List

@dataclasses.dataclass
class DataDefinition:
    hyper_cols: List[str]
    seed_col: str
    time_col: str
    environment_col: str
    algorithm_col: str


_data_def: DataDefinition | None = None

def data_definition(
    hyper_cols: Iterable[str],
    seed_col: str = 'seed',
    time_col: str = 'frame',
    environment_col: str | None = None,
    algorithm_col: str | None = None,

    make_global: bool = False,
) -> DataDefinition:
    d = DataDefinition(
        hyper_cols=list(sorted(hyper_cols)),
        seed_col=seed_col,
        time_col=time_col,
        environment_col=environment_col,
        algorithm_col=algorithm_col,
    )

    if make_global:
        global _data_def
        _data_def = d

    return d


def get_global_data_def():
    global _data_def
    assert _data_def is not None
    return _data_def


def maybe_global(d: DataDefinition | None):
    if d is None:
        return get_global_data_def()

    return d
