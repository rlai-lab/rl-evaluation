import copy
import numpy as np
import pandas as pd

from typing import Dict

import RlEvaluation.temporal as Temporal
import RlEvaluation.hypers as Hypers
import RlEvaluation.intervals as Intervals

# TODO:
#  - Allow specifying a "run" or "seed" column, then manipulate the data to match my desired shapes
#  - Allow specifying an "environment" or "problem" column

class ResultData:
    def __init__(self, data: pd.DataFrame):
        # we are going to be mutating the structure of the dataframe, so let's make our own copy
        # we will avoid mutating the underlying data
        self._data = data.copy()

        # evaluation configuration settings
        self._time_summaries: Dict[str, Temporal.TimeSummary] = {}
        self._metric_preference: Dict[str, Hypers.Preference] = {}

    # -------------------
    # -- Configuration --
    # -------------------
    def set_temporal_summary(self, metric: str, reducer: Temporal.TimeSummary):
        self._time_summaries[metric] = reducer

    def set_metric_preference(self, metric: str, preference: Hypers.Preference):
        self._metric_preference[metric] = preference

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
        return Hypers.select_best_hypers(self._data, f'{metric}.summary', self._metric_preference[metric], statistic)

    def get_learning_curve(self, metric: str, idx: int):
        data = self._data[metric].iloc[idx]
        data = np.asarray(data)
        return Intervals.bootstrap(data)

    # --------------
    # -- Internal --
    # --------------
    def _get_temporal_summary(self, metric: str):
        key = f'{metric}.summary'

        if key in self._data:
            return self._data[key]

        # detect if the data isn't temporal in nature
        if len(self._data[metric].iloc[0].shape) == 1:
            self._data[key] = self._data[metric]
            return self._data[key]

        # TODO: make this a real exception
        assert metric in self._time_summaries, 'Did not specify a strategy to summarize over time'
        reducer = self._time_summaries[metric]

        summary = self._data[metric].apply(reducer)
        self._data[key] = summary
        return summary
