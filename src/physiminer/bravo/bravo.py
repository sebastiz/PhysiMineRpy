"""
PhysiMiner - BRAVO Wireless pH Capsule Module
Cleaning, labelling, and Lyon Consensus 2.0 classification of BRAVO data.
"""

import pandas as pd
import numpy as np
import re
from typing import Optional


def data_bravo_clean(df: pd.DataFrame, date_col: str = "VisitDate") -> pd.DataFrame:
    """
    Clean and deduplicate a merged BRAVO export dataframe.

    Coerces pH/SAP/SI columns to numeric, parses the visit date,
    and deduplicates within each study.

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged BRAVO dataframe.
    date_col : str
        Name of the visit date column.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with one row per study.
    """
    df = df.copy()

    # Parse date
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Coerce pH/SAP/SI columns to numeric (skip date columns)
    numeric_patterns = ["pH", "SAP", "SI", "Refl", "Acid", "AET", "Day"]
    for col in df.columns:
        if "date" in col.lower():
            continue
        if any(pat.lower() in col.lower() for pat in numeric_patterns):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Deduplicate: keep first occurrence per study ID
    id_cols = [c for c in df.columns if "id" in c.lower() or "Id" in c]
    if id_cols:
        df = df.drop_duplicates(subset=[id_cols[0]], keep="first")

    df = df.reset_index(drop=True)
    return df


def data_bravo_day_labeller(
    df: pd.DataFrame,
    id_col: str = "HospNum_Id",
    date_col: str = "VisitDate",
) -> pd.DataFrame:
    """
    Standardise per-day AET column names to bravoDay1 … bravoDay6.

    Sierra exports use inconsistent naming (ReflDay1, ReflDay1_2, etc.).
    This function maps those to standardised columns and computes bravoNDays.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned BRAVO dataframe.
    id_col : str
        Patient/study identifier column.
    date_col : str
        Visit date column.

    Returns
    -------
    pd.DataFrame
        Dataframe with bravoDay1–bravoDay6 and bravoNDays added.
    """
    df = df.copy()

    # Patterns Sierra uses for per-day AET columns
    day_patterns = [
        # (day_number, list of possible column name patterns)
        (1, ["ReflDay1", "Day1AET", "day1", "Day1"]),
        (2, ["ReflDay2", "Day2AET", "day2", "Day2"]),
        (3, ["ReflDay3", "Day3AET", "day3", "Day3"]),
        (4, ["ReflDay4", "Day4AET", "day4", "Day4"]),
        (5, ["ReflDay5", "Day5AET", "day5", "Day5"]),
        (6, ["ReflDay6", "Day6AET", "day6", "Day6"]),
    ]

    for day_num, patterns in day_patterns:
        target_col = f"bravoDay{day_num}"
        if target_col not in df.columns:
            matched = None
            for pat in patterns:
                # Case-insensitive pattern search
                candidates = [c for c in df.columns if pat.lower() in c.lower()]
                if candidates:
                    matched = candidates[0]
                    break
            if matched:
                df[target_col] = pd.to_numeric(df[matched], errors="coerce")
            else:
                df[target_col] = np.nan

    # Count days with a recorded AET value
    day_cols = [f"bravoDay{i}" for i in range(1, 7)]
    df["bravoNDays"] = df[day_cols].notna().sum(axis=1)

    return df


def data_bravo_symptoms(df: pd.DataFrame, symp_col: str = "Symptoms") -> pd.DataFrame:
    """
    Parse BRAVO symptom free text into structured columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with a symptoms text column.
    symp_col : str
        Name of the free-text symptoms column.

    Returns
    -------
    pd.DataFrame
        Dataframe with AllSymps_BRAVO, AllSymps_BRAVOgrouped,
        AllSymps_BRAVOcompartment added.
    """
    df = df.copy()

    oesophageal_terms = [
        "heartburn", "regurgitation", "dysphagia", "chest pain",
        "odynophagia", "belching", "globus",
    ]
    lpr_terms = [
        "cough", "throat", "hoarseness", "laryngeal", "voice",
        "post nasal", "postnasal", "lpr",
    ]

    def parse_symptoms(text: str):
        if pd.isna(text) or str(text).strip() == "":
            return "", "Other", "Other"
        text_lower = str(text).lower()
        found = []
        for t in oesophageal_terms + lpr_terms:
            if t in text_lower:
                found.append(t.title())
        unique_symps = ", ".join(sorted(set(found))) if found else ""

        has_oeso = any(t in text_lower for t in oesophageal_terms)
        has_lpr = any(t in text_lower for t in lpr_terms)

        if has_oeso and has_lpr:
            grouped = "Mixed"
            compartment = "Mixed"
        elif has_oeso:
            grouped = "Oesophageal"
            compartment = "Oesophageal"
        elif has_lpr:
            grouped = "LPR"
            compartment = "Laryngopharyngeal"
        else:
            grouped = "Other"
            compartment = "Other"

        return unique_symps, grouped, compartment

    if symp_col in df.columns:
        parsed = df[symp_col].apply(parse_symptoms)
        df["AllSymps_BRAVO"] = parsed.apply(lambda x: x[0])
        df["AllSymps_BRAVOgrouped"] = parsed.apply(lambda x: x[1])
        df["AllSymps_BRAVOcompartment"] = parsed.apply(lambda x: x[2])
    else:
        df["AllSymps_BRAVO"] = ""
        df["AllSymps_BRAVOgrouped"] = "Other"
        df["AllSymps_BRAVOcompartment"] = "Other"

    return df


def gord_acid_bravo_lyon(
    df: pd.DataFrame,
    pathological_aet: float = 6.0,
    normal_aet: float = 4.0,
    min_positive_days: int = 2,
) -> pd.DataFrame:
    """
    Apply Lyon Consensus 2.0 three-tier classification to BRAVO data.

    Parameters
    ----------
    df : pd.DataFrame
        Labelled BRAVO dataframe with bravoDay1–bravoDay6 and bravoNDays.
    pathological_aet : float
        AET threshold for conclusive GORD (default 6%).
    normal_aet : float
        AET threshold below which GORD is excluded (default 4%).
    min_positive_days : int
        Minimum number of days above pathological_aet for conclusive GORD.

    Returns
    -------
    pd.DataFrame
        Dataframe with AcidRefluxBRAVO_Lyon, AcidRefluxBRAVO,
        AcidRefluxBRAVOTotalOnly, AcidRefluxBRAVOAv added.
    """
    df = df.copy()

    day_cols = [f"bravoDay{i}" for i in range(1, 7)]
    available_day_cols = [c for c in day_cols if c in df.columns]

    def classify_row(row):
        days = pd.to_numeric(
            pd.Series([row.get(c, np.nan) for c in available_day_cols]),
            errors="coerce"
        ).dropna()

        if len(days) == 0:
            return "Inconclusive", 0, 0, 0

        n_positive = (days > pathological_aet).sum()
        all_normal = (days < normal_aet).all()
        average_aet = days.mean()

        # Per-day conclusive GORD
        if n_positive >= min_positive_days:
            lyon_class = "Conclusive GORD"
            acid_reflux = 1
        elif all_normal:
            lyon_class = "GORD excluded"
            acid_reflux = 0
        else:
            lyon_class = "Inconclusive"
            acid_reflux = 0

        # Average-day analysis
        acid_reflux_av = 1 if average_aet > pathological_aet else 0

        # Total-only (single-day worst approach, but using average below normal)
        acid_reflux_total_only = 1 if all_normal else 0
        if acid_reflux_total_only == 1:
            acid_reflux_total_only = 0  # inverted: 1 means normal by total
        else:
            acid_reflux_total_only = 0

        # Simpler: TotalOnly flags patients where average < normal
        acid_reflux_total_only = 1 if (average_aet < normal_aet) else 0

        return lyon_class, acid_reflux, acid_reflux_total_only, acid_reflux_av

    results = df.apply(classify_row, axis=1)
    df["AcidRefluxBRAVO_Lyon"] = results.apply(lambda x: x[0])
    df["AcidRefluxBRAVO"] = results.apply(lambda x: x[1])
    df["AcidRefluxBRAVOTotalOnly"] = results.apply(lambda x: x[2])
    df["AcidRefluxBRAVOAv"] = results.apply(lambda x: x[3])

    return df


def gord_bravo_wda_and_average(
    df: pd.DataFrame,
    n_days: int = 4,
    aet_threshold: float = 6.0,
) -> pd.DataFrame:
    """
    Worst-day and average-day AET analysis for BRAVO data.

    Parameters
    ----------
    df : pd.DataFrame
        Classified BRAVO dataframe.
    n_days : int
        Number of recording days (typically 4).
    aet_threshold : float
        AET threshold for positivity.

    Returns
    -------
    pd.DataFrame
        Dataframe with worstt, average, worstDaypH,
        NumDaysBravoPositive, WorstDayAnalysisGORDPositive added.
    """
    df = df.copy()

    day_cols = [f"bravoDay{i}" for i in range(1, n_days + 1)]
    available = [c for c in day_cols if c in df.columns]

    day_data = df[available].apply(pd.to_numeric, errors="coerce")

    df["worstt"] = day_data.max(axis=1)
    df["average"] = day_data.mean(axis=1)
    df["worstDaypH"] = day_data.idxmax(axis=1).str.extract(r"(\d+)$").astype(float)
    df["NumDaysBravoPositive"] = (day_data >= aet_threshold).sum(axis=1)
    df["WorstDayAnalysisGORDPositive"] = (df["worstt"] > aet_threshold).astype(int)

    return df
