import unittest
import numpy as np
import pandas as pd

from RlEvaluation._utils.data import normalizeDataType, make_wide_format, is_wide_format
from tests.test_utils.mock_data import generate_split_over_seed

class TestData(unittest.TestCase):
    def test_normalizeDataType(self):
        # turn pandas dataframe into a numpy array
        test_data = pd.DataFrame({
            'alpha': [0.01, 0.01, 0.1],
            'results': [1, 2, 3],
        })
        got = normalizeDataType(test_data, 2, 'results')
        self.assertIsInstance(got, np.ndarray)
        self.assertEqual(np.ndim(got), 2)

        # keep numpy array untouched
        test_data = np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
        ])
        got = normalizeDataType(test_data, 2)
        self.assertIsInstance(got, np.ndarray)
        self.assertEqual(np.ndim(got), 2)

        # TODO: test shape normalization

    def test_make_wide_format(self):
        # works for one results column
        df = generate_split_over_seed()

        hypers = {'stepsize', 'optimizer'}
        metrics = {'results'}

        got = make_wide_format(df, hypers=hypers, metrics=metrics, seed_col='run')

        self.assertEqual(len(got), 6)
        self.assertEqual(got.iloc[0]['results'].shape, (10, 300))

        # works for two results columns
        df2 = df.copy()
        df2['results-2'] = df2['results'] * 2
        metrics = {'results', 'results-2'}

        got = make_wide_format(df2, hypers=hypers, metrics=metrics, seed_col='run')

        self.assertEqual(len(got), 6)
        self.assertEqual(got.iloc[0]['results'].shape, (10, 300))
        self.assertEqual(got.iloc[0]['results-2'].shape, (10, 300))

        # should not change already wide data
        self.assertFalse(is_wide_format(df, metrics, 'run'))
        self.assertFalse(is_wide_format(df2, metrics, 'run'))

        got = make_wide_format(df2, hypers=hypers, metrics=metrics, seed_col='run')
        self.assertTrue(is_wide_format(got, metrics, 'run'))

        got2 = make_wide_format(got, hypers=hypers, metrics=metrics, seed_col='run')
        self.assertEqual(id(got), id(got2))
