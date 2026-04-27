"""
PhysiMiner - Cross-Test Merging Module
Merges BRAVO, impedance, and HRM datasets by patient and closest visit date.
"""

import pandas as pd
import numpy as np
from typing import Literal, Optional


def test_merge(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    id_col: str = "HospNum_Id",
    date_col1: Optional[str] = None,
    date_col2: Optional[str] = None,
    join_type: Literal["inner", "left", "outer"] = "inner",
    max_days_apart: Optional[int] = None,
) -> pd.DataFrame:
    """
    Merge two test datasets by patient, selecting the pair closest in time.

    For patients with multiple studies in one dataset, the study closest
    to the other test's date is retained.

    Parameters
    ----------
    df1 : pd.DataFrame
        First test dataset.
    df2 : pd.DataFrame
        Second test dataset.
    id_col : str
        Patient identifier column present in both dataframes.
    date_col1 : str, optional
        Date column in df1. Auto-detected if None.
    date_col2 : str, optional
        Date column in df2. Auto-detected if None.
    join_type : str
        Type of join: 'inner', 'left', or 'outer'.
    max_days_apart : int, optional
        Maximum allowable days between paired studies. Pairs exceeding this
        are dropped.

    Returns
    -------
    pd.DataFrame
        One row per patient with columns from both datasets and Date_ABS_Diff.
    """
    df1 = df1.copy()
    df2 = df2.copy()

    # Auto-detect date columns
    if date_col1 is None:
        date_col1 = _find_date_col(df1)
    if date_col2 is None:
        date_col2 = _find_date_col(df2)

    # Ensure date columns are datetime
    if date_col1 and date_col1 in df1.columns:
        df1[date_col1] = pd.to_datetime(df1[date_col1], errors="coerce")
    if date_col2 and date_col2 in df2.columns:
        df2[date_col2] = pd.to_datetime(df2[date_col2], errors="coerce")

    # Suffix management to avoid collision
    suffix1 = ".x" if date_col1 == date_col2 else ""
    suffix2 = ".y" if date_col1 == date_col2 else ""

    merged = df1.merge(df2, on=id_col, how="outer", suffixes=(suffix1, suffix2))

    # Compute date difference
    col1_final = date_col1 + suffix1 if suffix1 else date_col1
    col2_final = date_col2 + suffix2 if suffix2 else date_col2

    if col1_final in merged.columns and col2_final in merged.columns:
        diff = (
            merged[col1_final].dt.normalize() - merged[col2_final].dt.normalize()
        ).abs()
        merged["Date_ABS_Diff"] = diff.dt.days
    else:
        merged["Date_ABS_Diff"] = np.nan

    # For each patient, keep only the pair with minimum date difference
    if "Date_ABS_Diff" in merged.columns:
        merged = (
            merged.sort_values("Date_ABS_Diff")
            .groupby(id_col, as_index=False)
            .first()
        )

    # Apply join type filter
    if join_type == "inner":
        # Drop rows where either test is missing
        left_only = df1[id_col].unique()
        right_only = df2[id_col].unique()
        both = set(left_only) & set(right_only)
        merged = merged[merged[id_col].isin(both)]
    elif join_type == "left":
        merged = merged[merged[id_col].isin(df1[id_col].unique())]

    # Apply max_days_apart filter
    if max_days_apart is not None and "Date_ABS_Diff" in merged.columns:
        merged = merged[
            merged["Date_ABS_Diff"].isna() | (merged["Date_ABS_Diff"] <= max_days_apart)
        ]

    return merged.reset_index(drop=True)


def triple_test_merge(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    id_col: str = "HospNum_Id",
    join_type: Literal["inner", "left", "outer"] = "inner",
    max_days_apart: Optional[int] = None,
) -> pd.DataFrame:
    """
    Merge three test datasets by closest visit date.

    Performs sequential pairwise merges: (df1 + df2) then + df3.

    Parameters
    ----------
    df1 : pd.DataFrame
        First test dataset (e.g. HRM).
    df2 : pd.DataFrame
        Second test dataset (e.g. impedance).
    df3 : pd.DataFrame
        Third test dataset (e.g. BRAVO).
    id_col : str
        Patient identifier column.
    join_type : str
        'inner', 'left', or 'outer'.
    max_days_apart : int, optional
        Maximum allowable days between any paired studies.

    Returns
    -------
    pd.DataFrame
        One row per patient with columns from all three datasets.
    """
    merged_12 = test_merge(
        df1, df2, id_col=id_col, join_type=join_type,
        max_days_apart=max_days_apart,
    )
    merged_123 = test_merge(
        merged_12, df3, id_col=id_col, join_type=join_type,
        max_days_apart=max_days_apart,
    )
    return merged_123


def _find_date_col(df: pd.DataFrame) -> Optional[str]:
    """Heuristically find the primary date column in a dataframe."""
    for col in df.columns:
        if "VisitDate" in col or col == "VisitDate":
            return col
    for col in df.columns:
        if "date" in col.lower() and df[col].dtype in ["datetime64[ns]", "object"]:
            return col
    return None
