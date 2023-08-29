import unittest
import pandas as pd

from RlEvaluation.hypers import select_best_hypers, Preference
from RlEvaluation.config import data_definition

class TestHypers(unittest.TestCase):
    def test_select_best_hypers(self):
        test_data = pd.DataFrame({
            'alpha': [0.1, 0.01, 0.001],
            'seed': [0, 0, 0],
            'result': [0, 2, 1],
        })

        d = data_definition(hyper_cols=['alpha'])

        best = select_best_hypers(test_data, 'result', Preference.high, data_definition=d)
        self.assertEqual(best.best_configuration[0], 0.01)
