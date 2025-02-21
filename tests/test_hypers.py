import polars as pl

from rlevaluation.hypers import select_best_hypers, Preference
from rlevaluation.config import data_definition

def test_select_best_hypers():
    test_data = pl.DataFrame({
        'alpha': [0.1, 0.01, 0.001, 0.01],
        'seed': [0, 0, 0, 1],
        'result': [0, 2, 1, 4],
    })

    d = data_definition(hyper_cols=['alpha'])

    best = select_best_hypers(test_data, 'result', Preference.high, data_definition=d)
    assert best.best_configuration['alpha'] == 0.01
