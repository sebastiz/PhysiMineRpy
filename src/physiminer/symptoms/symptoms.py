"""
PhysiMiner - Symptom Classification and Data Quality Module
SAP/SI classification, free-text symptom extraction, and completeness filtering.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Optional


# ── Symptom Association Classification ──────────────────────────────────────

def classify_symptom_association(
    df: pd.DataFrame,
    sap_threshold: float = 95.0,
    si_threshold: float = 50.0,
) -> pd.DataFrame:
    """
    Binary SAP/SI classification per Lyon Consensus 2.0.

    Adds SAP_positive_* and SI_positive_* binary columns per symptom,
    plus summary columns AnySAP_positive, AnySI_positive, SymptomAssoc_positive.

    Parameters
    ----------
    df : pd.DataFrame
        Impedance dataframe with SAP and SI columns.
    sap_threshold : float
        SAP threshold for positive association (default 95%).
    si_threshold : float
        SI threshold for positive association (default 50%).

    Returns
    -------
    pd.DataFrame
        Dataframe with symptom association columns added.
    """
    df = df.copy()

    sap_cols = [c for c in df.columns if c.startswith("SAP")]
    si_cols = [c for c in df.columns if c.startswith("SI")]

    sap_positive_cols = []
    for col in sap_cols:
        positive_col = f"SAP_positive_{col}"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[positive_col] = (df[col] >= sap_threshold).astype(int)
        sap_positive_cols.append(positive_col)

    si_positive_cols = []
    for col in si_cols:
        positive_col = f"SI_positive_{col}"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[positive_col] = (df[col] >= si_threshold).astype(int)
        si_positive_cols.append(positive_col)

    df["AnySAP_positive"] = (
        df[sap_positive_cols].any(axis=1).astype(int)
        if sap_positive_cols else 0
    )
    df["AnySI_positive"] = (
        df[si_positive_cols].any(axis=1).astype(int)
        if si_positive_cols else 0
    )
    df["SymptomAssoc_positive"] = (
        (df["AnySAP_positive"] == 1) | (df["AnySI_positive"] == 1)
    ).astype(int)

    return df


# ── Free-Text Symptom Extraction ─────────────────────────────────────────────

# Canonical symptom definitions with keyword aliases
_SYMPTOM_DEFINITIONS = {
    "Heartburn": [
        "heartburn", "pyrosis", "burning chest", "acid", "reflux",
    ],
    "Regurgitation": [
        "regurgitation", "regurg", "bringing up", "vomiting food",
    ],
    "Dysphagia": [
        "dysphagia", "difficulty swallowing", "trouble swallowing",
        "food stick", "food getting stuck",
    ],
    "ChestPain": [
        "chest pain", "chest discomfort", "chest tightness",
        "retrosternal pain",
    ],
    "Belching": [
        "belch", "belching", "burp", "eructation",
    ],
    "Cough": [
        "cough", "coughing", "chronic cough",
    ],
    "ThroatSymptoms": [
        "throat", "throat clearing", "throat pain", "sore throat",
        "globus", "lump in throat",
    ],
    "Nausea": [
        "nausea", "nauseous", "feeling sick", "queasiness",
    ],
    "Vomiting": [
        "vomiting", "vomit", "emesis",
    ],
    "StomachPain": [
        "stomach pain", "abdominal pain", "epigastric pain",
        "belly pain", "gastric pain",
    ],
}


def extract_symptoms(
    df: pd.DataFrame,
    text_cols: "str | List[str]" = "ClinicalNote",
) -> pd.DataFrame:
    """
    Extract symptom presence from free-text clinical note columns.

    Adds binary Sx_{SymptomName} columns for each canonical symptom.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with free-text columns.
    text_cols : str or list of str
        Column name(s) containing free text.

    Returns
    -------
    pd.DataFrame
        Dataframe with Sx_* binary columns added.
    """
    df = df.copy()

    if isinstance(text_cols, str):
        text_cols = [text_cols]

    # Combine all text columns into one for searching
    def combine_text(row):
        parts = []
        for col in text_cols:
            if col in row.index and not pd.isna(row[col]):
                parts.append(str(row[col]).lower())
        return " ".join(parts)

    combined_text = df.apply(combine_text, axis=1)

    for symptom_name, keywords in _SYMPTOM_DEFINITIONS.items():
        col_name = f"Sx_{symptom_name}"
        pattern = "|".join(re.escape(kw) for kw in keywords)
        df[col_name] = combined_text.str.contains(
            pattern, case=False, regex=True, na=False
        ).astype(int)

    return df


def symptom_burden_summary(
    df: pd.DataFrame,
    sx_prefix: str = "Sx_",
) -> pd.DataFrame:
    """
    Tally and categorise symptom burden per patient.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with Sx_* binary symptom columns.
    sx_prefix : str
        Prefix identifying symptom columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with nSymptoms and symptomBurden columns added.
    """
    df = df.copy()

    sx_cols = [c for c in df.columns if c.startswith(sx_prefix)]

    if not sx_cols:
        df["nSymptoms"] = 0
        df["symptomBurden"] = "None"
        return df

    df["nSymptoms"] = df[sx_cols].sum(axis=1)

    def burden_label(n):
        if n == 0:
            return "None"
        elif n <= 2:
            return "Moderate"
        elif n <= 4:
            return "High"
        else:
            return "Very High"

    df["symptomBurden"] = df["nSymptoms"].apply(burden_label)

    return df


# ── Data Quality Filtering ────────────────────────────────────────────────────

def filter_by_completeness(
    df: pd.DataFrame,
    col_complete_threshold: float = 0.9,
    row_complete_threshold: Optional[float] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Remove columns and optionally rows with insufficient complete data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col_complete_threshold : float
        Minimum proportion of non-missing values for a column to be retained
        (default 0.9 = 90%).
    row_complete_threshold : float, optional
        Minimum proportion of non-missing values for a row to be retained.
        If None, no row filtering is applied.
    verbose : bool
        If True, print names of dropped columns.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    df = df.copy()
    n_rows = len(df)

    # Column-level filtering
    completeness = df.notna().mean()
    cols_to_drop = completeness[completeness < col_complete_threshold].index.tolist()

    if verbose and cols_to_drop:
        print(
            f"Dropping {len(cols_to_drop)} column(s) with "
            f"< {col_complete_threshold * 100:.0f}% complete data:"
        )
        for col in cols_to_drop:
            print(f"  {col}")

    df = df.drop(columns=cols_to_drop)

    # Row-level filtering
    if row_complete_threshold is not None:
        row_completeness = df.notna().mean(axis=1)
        n_before = len(df)
        df = df[row_completeness >= row_complete_threshold]
        if verbose:
            n_dropped = n_before - len(df)
            if n_dropped > 0:
                print(
                    f"Dropping {n_dropped} row(s) with "
                    f"< {row_complete_threshold * 100:.0f}% complete data"
                )

    return df.reset_index(drop=True)
