import copy
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import cast, Any, Dict, Optional, Sequence

import RlEvaluation._utils.data as du

import RlEvaluation.hypers as Hypers
import RlEvaluation.temporal as Temporal
import RlEvaluation.intervals as Intervals

@dataclass
class Metric:
    time_summary: Temporal.TimeSummary = Temporal.mean
    preference: Hypers.Preference = Hypers.Preference.high

# TODO:
#  - Allow specifying a "run" or "seed" column, then manipulate the data to match my desired shapes
#  - Allow specifying an "environment" or "problem" column

class ResultData:
    def __init__(self, data: pd.DataFrame, config: Dict[str, Metric], hyperparam_cols: Optional[Sequence[str]] = None, run_col: str = 'run'):
        # we are going to be mutating the structure of the dataframe, so let's make our own copy
        # we will avoid mutating the underlying data
        self._data = data.copy()

        # data format
        self._metrics = set(config.keys())
        self._run = run_col

        if hyperparam_cols is None:
            self._hypers = du.get_other_cols(self._data, self._metrics, self._run)
        else:
            self._hypers = set(hyperparam_cols)

        # evaluation configuration settings
        self._c = config

        # preliminary data prep
        self._prep_data()

    # --------------
    # -- External --
    # --------------
    def split_over_hyper(self, hyper: str):
        vals = self._data[hyper].unique()

        for v in vals:
            res = copy.copy(self)
            res._data = self._data[self._data[hyper] == v].copy()

            yield v, res

    def get_best_hyper_idx(self, metric: str, statistic=np.mean):
        self._get_temporal_summary(metric)
        pref = self._c[metric].preference
        return Hypers.select_best_hypers(self._data, f'{metric}.summary', pref, statistic)

    def get_learning_curve(self, metric: str, idx: int):
        data = self._data[metric].iloc[idx]
        data = np.asarray(data)
        return Intervals.bootstrap(data)

    # --------------
    # -- Internal --
    # --------------
    def _prep_data(self):
        # if not wide format, then convert
        self._data = du.make_wide_format(self._data, self._hypers, self._metrics, self._run)

    def _get_temporal_summary(self, metric: str):
        key = f'{metric}.summary'

        if key in self._data:
            return self._data[key]

        # detect if the data isn't temporal in nature
        if len(self._data[metric].iloc[0].shape) == 1:
            self._data[key] = self._data[metric]
            return self._data[key]

        # TODO: make this a real exception
        assert metric in self._c, 'Unknown metric'
        reducer = self._c[metric].time_summary

        # TODO: pandas types are pretty bad right now
        reducer = cast(Any, reducer)
        summary = self._data[metric].apply(reducer)
        self._data[key] = summary
        return summary
