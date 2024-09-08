from rlevaluation.config import DataDefinition
from tests.test_utils.mock_data import learning_curve_data

from rlevaluation.temporal import extract_learning_curves

def test_extract_learning_curves1():
    df = learning_curve_data()

    got_x, got_y = extract_learning_curves(
        df,
        { 'alpha': 0.01, 'beta': 4, 'gamma': 0.99 },
        metric='metric',
        data_definition=DataDefinition(
            hyper_cols=['alpha', 'beta', 'gamma'],
            seed_col='seed',
            time_col='time',
        )
    )

    assert len(got_x[0]) == 25
    assert len(got_y[0]) == 25

    assert len(got_x) == 10
    assert len(got_y) == 10
