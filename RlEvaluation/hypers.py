import pandas as pd
import enum
from RlEvaluation._utils.numba import over_first_axis
from RlEvaluation._utils.data import Reducer, normalizeDataType, compileReducer
from RlEvaluation.utils.math import mean
from RlEvaluation.tools import subsetDF

class Preference(enum.Enum):
    high = 'high'
    low = 'low'

def select_best_hypers(df: pd.DataFrame, column: str, prefer: Preference, reducer: Reducer = mean):
    f = compileReducer(reducer)
    data = normalizeDataType(df, 2, col=column)
    # TODO: report uncertainty in hyper selection
    reduced = over_first_axis(data, f)

    if prefer == Preference.high:
        best_idx = reduced.argmax()
    elif prefer == Preference.low:
        best_idx = reduced.argmin()
    else:
        raise UnknownPreferenceException()

    idx = int(best_idx)
    return idx

def sliceOverHyper(df: pd.DataFrame, hyper: str):
    values = df[hyper].unique()

    for v in values:
        sub = subsetDF(df, { hyper: v })
        yield v, sub.reset_index()


class UnknownPreferenceException(Exception):
    ...
