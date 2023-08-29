import pandas as pd

from RlEvaluation.config import DataDefinition, maybe_global

def step_weighted_return(df: pd.DataFrame, return_col: str = 'return', d: DataDefinition | None = None):
    d = maybe_global(d)
    max_steps = df[d.time_col].max()
    return df[return_col] * (df[d.time_col] / max_steps)
