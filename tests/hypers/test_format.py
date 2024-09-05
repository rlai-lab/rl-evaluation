import polars as pl
import polars.testing as plt
from rlevaluation.hypers.format import is_score_format, make_score_format
from rlevaluation.config import data_definition
from rlevaluation.temporal import TimeSummary

# ---------------------
# -- is_score_format --
# ---------------------

def test_is_score_format1():
    df = pl.DataFrame({
        'alpha': [0.1, 0.01, 0.001],
        'seed': [0, 0, 0],
        'result': [0, 2, 1],
    })

    d = data_definition(hyper_cols=['alpha'])
    assert is_score_format(df, 'result', d)

def test_is_score_format2():
    df = pl.DataFrame({
        'alpha': [0.1, 0.01, 0.001],
        'seed': [0, 0, 0],
        'result': [[0], [2], [1]],
    })

    d = data_definition(hyper_cols=['alpha'])
    assert not is_score_format(df, 'result', d)

def test_is_score_format3():
    df = pl.DataFrame({
        'alpha': [0.1, 0.01, 0.001, 0.001],
        'seed': [0, 0, 0, 1],
        'result': [0, 1, 2, 1],
    })

    d = data_definition(hyper_cols=['alpha'])
    assert is_score_format(df, 'result', d)

def test_is_score_format4():
    df = pl.DataFrame({
        'alpha': [0.1, 0.1, 0.1],
        'time': [1, 2, 3],
        'seed': [0, 0, 0],
        'result': [0, 1, 2],
    })

    d = data_definition(hyper_cols=['alpha'])
    assert not is_score_format(df, 'result', d)

# -----------------------
# -- make_score_format --
# -----------------------

def test_make_score_format1():
    df = pl.DataFrame({
        'alpha': [0.1, 0.1, 0.1, 0.2],
        'time': [1, 2, 3, 1],
        'seed': [0, 0, 0, 0],
        'result': [0., 1, 2, 4],
    })

    d = data_definition(
        hyper_cols=['alpha'],
        time_col='time',
        seed_col='seed',
    )

    score_df = make_score_format(
        df,
        metric='result',
        dd=d,
        time_summary=TimeSummary.mean,
    )

    expected = pl.DataFrame({
        'alpha': [0.1, 0.2],
        'seed': [0, 0],
        'result': [1.0, 4.0],
    })

    plt.assert_frame_equal(
        score_df,
        expected,
        check_column_order=False,
        check_row_order=False,
    )
