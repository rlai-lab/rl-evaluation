import itertools
import numpy as np
import pandas as pd

from typing import Any, Dict, Sequence


def split_over_column(df: pd.DataFrame, col: str):
    vals = df[col].unique()

    for v in vals:
        yield v, df[df[col] == v]


def split_over_columns(df: pd.DataFrame, cols: Sequence[str]):
    uniques = [ df[col].unique() for col in cols ]

    for vals in itertools.product(*uniques):
        mask = build_mask(df, cols, vals)
        sub = df[mask]

        yield vals, sub


def subset_df_by_dict(df: pd.DataFrame, conds: Dict[str, Any]):
    mask = build_mask_from_dict(df, conds)
    return df[mask].reset_index(drop=True)

def subset_df(df: pd.DataFrame, cols: Sequence[str], vals: Sequence[Any]):
    mask = build_mask(df, cols, vals)
    return df[mask].reset_index(drop=True)


def build_mask(df: pd.DataFrame, cols: Sequence[str], vals: Sequence[Any]):
    it = zip(cols, vals, strict=True)
    mask = np.ones(len(df), dtype=bool)
    for c, v in it:
        mask = mask & ((df[c] == v) | df[c].isna())

    return mask

def build_mask_from_dict(df: pd.DataFrame, conds: Dict[str, Any]):
    mask = np.ones(len(df), dtype=bool)
    for key, cond in conds.items():
        if isinstance(cond, dict):
            mask = mask | build_mask_from_dict(df, cond)

        elif isinstance(cond, list):
            mask = mask & (df[key].isin(cond))

        elif isinstance(cond, float) and np.isnan(cond):
            mask = mask & (df[key].isna())

        else:
            mask = mask & (df[key] == cond)

    return mask
