import unittest
import pandas as pd

from RlEvaluation.hypers import selectBestHypers, Preference

class TestHypers(unittest.TestCase):
    def test_selectBestHypers(self):
        test_data = pd.DataFrame({
            'alpha': [0.1, 0.01, 0.001],
            'result': [0, 1, 2],
        })

        best = selectBestHypers(test_data, 'result', Preference.high)
        self.assertIsInstance(best, pd.DataFrame)

        expected = pd.DataFrame({
            'alpha': 0.001,
            'result': 2,
        }, index=[2])
        pd.testing.assert_frame_equal(best, expected)
