import polars as pl

from rlevaluation.config import data_definition
from rlevaluation.hypers.utils import group_measurements

def test_group_measurements1():
    d = data_definition(
        hyper_cols=['alpha'],
    )

    df = pl.DataFrame({
        'alpha': [0.1, 0.01, 0.001],
        'seed': [0, 0, 0],
        'result': [0, 2, 1],
    })

    group = group_measurements(
        df,
        metric='result',
        data_definition=d,
    )

    assert group.n == 3
