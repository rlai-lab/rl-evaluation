import polars as pl
from rlevaluation.config import DataDefinition
from rlevaluation.temporal import TimeSummary

def is_score_format(df: pl.DataFrame, metric: str, dd: DataDefinition) -> bool:
    cols = set(dd.hyper_cols).intersection(df.columns)
    cols |= {dd.seed_col}

    for _, run_data in df.group_by(cols):
        if len(run_data) != 1:
            return False

        measures = run_data[metric]
        if not measures.dtype.is_numeric():
            return False

    return True


def make_score_format(
    df: pl.DataFrame,
    metric: str,
    dd: DataDefinition,
    time_summary: TimeSummary,
) -> pl.DataFrame:
    if is_score_format(df, metric, dd):
        return df

    cols = set(dd.hyper_cols).intersection(df.columns)
    cols |= {dd.seed_col}

    agg = time_summary_polars(time_summary)
    return df.group_by(cols).agg(agg(metric))


def time_summary_polars(ts: TimeSummary):
    if ts == TimeSummary.mean:
        return pl.mean

    elif ts == TimeSummary.sum:
        return pl.sum

    raise Exception('Unknown summary type')
