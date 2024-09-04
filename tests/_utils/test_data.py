import numpy as np
import pandas as pd

from rlevaluation._utils.data import normalizeDataType, make_wide_format, is_wide_format
from tests.test_utils.mock_data import generate_split_over_seed

def test_normalizeDataType():
    # turn pandas dataframe into a numpy array
    test_data = pd.DataFrame({
        'alpha': [0.01, 0.01, 0.1],
        'results': [1, 2, 3],
    })
    got = normalizeDataType(test_data, 2, 'results')
    assert isinstance(got, np.ndarray)
    assert np.ndim(got) == 2

    # keep numpy array untouched
    test_data = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
    ])
    got = normalizeDataType(test_data, 2)
    assert isinstance(got, np.ndarray)
    assert np.ndim(got) == 2

    # TODO: test shape normalization

def test_make_wide_format():
    # works for one results column
    df = generate_split_over_seed()

    hypers = {'stepsize', 'optimizer'}
    metrics = {'results'}

    got = make_wide_format(df, hypers=hypers, metrics=metrics, seed_col='run')

    assert len(got) == 6
    assert got.iloc[0]['results'].shape == (10, 300)

    # works for two results columns
    df2 = df.copy()
    df2['results-2'] = df2['results'] * 2
    metrics = {'results', 'results-2'}

    got = make_wide_format(df2, hypers=hypers, metrics=metrics, seed_col='run')

    assert len(got) == 6
    assert got.iloc[0]['results'].shape == (10, 300)
    assert got.iloc[0]['results-2'].shape == (10, 300)

    # should not change already wide data
    assert not is_wide_format(df, metrics, 'run')
    assert not is_wide_format(df2, metrics, 'run')

    got = make_wide_format(df2, hypers=hypers, metrics=metrics, seed_col='run')
    assert is_wide_format(got, metrics, 'run')

    got2 = make_wide_format(got, hypers=hypers, metrics=metrics, seed_col='run')
    assert id(got) == id(got2)
