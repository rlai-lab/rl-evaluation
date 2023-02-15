import numpy as np
import pandas as pd
from typing import Any, Dict

def subsetDF(df: pd.DataFrame, conds: Dict[str, Any]):
    mask = _buildMask(df, conds)
    return df[mask].reset_index(drop=True)

# ------------------------
# -- Internal Utilities --
# ------------------------
def _buildMask(df: pd.DataFrame, conds: Dict[str, Any]):
    mask = np.ones(len(df), dtype=bool)
    for key, cond in conds.items():
        if isinstance(cond, dict):
            mask = mask | _buildMask(df, cond)

        elif isinstance(cond, list):
            mask = mask & (df[key].isin(cond))

        else:
            mask = mask & (df[key] == cond)

    return mask
