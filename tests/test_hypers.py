import unittest
import pandas as pd

from RlEvaluation.hypers import select_best_hypers, Preference

class TestHypers(unittest.TestCase):
    def test_select_best_hypers(self):
        test_data = pd.DataFrame({
            'alpha': [0.1, 0.01, 0.001],
            'result': [0, 2, 1],
        })

        best = select_best_hypers(test_data, 'result', Preference.high)
        self.assertEqual(best, 1)
