import pandas as pd

from RlEvaluation.config import DataDefinition, maybe_global

def add_step_weighted_return(
    df: pd.DataFrame,
    episode_len_col: str = 'steps',
    return_col: str = 'return',
    d: DataDefinition | None = None,
):
    d = maybe_global(d)
    cols = _get_data_def_columns(df, d)

    def red(sub):
        return sub[return_col] * (sub[episode_len_col] / sub[d.time_col].max())

    groups = df.groupby(cols, dropna=False, as_index=False, group_keys=False)
    df['step_weighted_return'] = groups.apply(red)


def _get_data_def_columns(df: pd.DataFrame, d: DataDefinition):
    cols = []

    if d.algorithm_col in df.columns:
        cols.append(d.algorithm_col)

    if d.environment_col in df.columns:
        cols.append(d.environment_col)

    if d.seed_col in df.columns:
        cols.append(d.seed_col)

    cols += d.hyper_cols
    return cols
