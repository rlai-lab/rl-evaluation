import polars as pl

from rlevaluation.config import DataDefinition, maybe_global

def add_step_weighted_return(
    df: pl.DataFrame,
    episode_len_col: str = 'steps',
    return_col: str = 'return',
    d: DataDefinition | None = None,
):
    d = maybe_global(d)
    cols = _get_data_def_columns(df, d)

    def red(sub):
        return sub[return_col] * (sub[episode_len_col] / sub[d.time_col].max())

    groups = df.group_by(cols)
    df['step_weighted_return'] = groups.map_groups(red)


def _get_data_def_columns(df: pl.DataFrame, d: DataDefinition):
    cols = []

    if d.algorithm_col in df.columns:
        cols.append(d.algorithm_col)

    if d.environment_col in df.columns:
        cols.append(d.environment_col)

    if d.seed_col in df.columns:
        cols.append(d.seed_col)

    for col in d.hyper_cols:
        if col in df:
            cols.append(col)

    return cols
