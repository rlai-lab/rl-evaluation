import unittest
import numpy as np
import pandas as pd

from RlEvaluation._utils.data import normalizeDataType

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
