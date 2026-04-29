"""
PhysiMiner - Ambulatory pH-Impedance Monitoring Module
Cleaning, symptom categorisation, and Lyon Consensus 2.0 classification.
"""

import pandas as pd
import numpy as np
from typing import Optional


def data_imp_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a merged pH-impedance dataframe.

    Coerces reflux metric columns to numeric, converts dates, and creates
    binary Sx_* indicators for each symptom type based on non-missing SAP/SI.

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged impedance + symptom dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = df.copy()

    # Parse date columns
    for col in df.columns:
        if "date" in col.lower() or "Date" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Coerce reflux metric columns
    numeric_patterns = [
        "AET", "Acid", "Refl", "SAP", "SI", "Episode", "MNBI",
        "pH", "impedance", "Imp", "Total", "Percent",
    ]
    for col in df.columns:
        if any(pat.lower() in col.lower() for pat in numeric_patterns):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create binary Sx_* indicators from SAP columns
    sap_cols = [c for c in df.columns if c.upper().startswith("SAP")]
    for col in sap_cols:
        symptom_name = col.replace("SAP", "Sx_").replace("sap", "Sx_")
        df[symptom_name] = df[col].notna().astype(int)

    return df.reset_index(drop=True)


def data_imp_symptoms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorise impedance symptoms by anatomical group.

    Creates SAPOesophageal, SAPLPR, AllSymps_Impgrouped, AllImpSymptom.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned impedance dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with symptom group columns added.
    """
    df = df.copy()

    oesophageal_keywords = [
        "heartburn", "regurgitation", "regurg", "chestpain", "chest_pain",
        "belch", "dysphagia", "odynophagia",
    ]
    lpr_keywords = ["cough", "throat", "hoarse", "lpr", "laryngeal", "voice"]

    # Find SAP columns for each group
    sap_cols = [c for c in df.columns if "SAP" in c or "sap" in c.lower()]

    def get_group_sap(row, keywords):
        vals = []
        for col in sap_cols:
            if any(kw in col.lower() for kw in keywords):
                v = pd.to_numeric(row.get(col, np.nan), errors="coerce")
                if not pd.isna(v):
                    vals.append(v)
        return max(vals) if vals else np.nan

    df["SAPOesophageal"] = df.apply(
        lambda r: get_group_sap(r, oesophageal_keywords), axis=1
    )
    df["SAPLPR"] = df.apply(lambda r: get_group_sap(r, lpr_keywords), axis=1)

    def classify_group(row):
        has_oeso = not pd.isna(row.get("SAPOesophageal", np.nan))
        has_lpr = not pd.isna(row.get("SAPLPR", np.nan))
        if has_oeso and has_lpr:
            return "Mixed"
        elif has_oeso:
            return "Oesophageal"
        elif has_lpr:
            return "LPR"
        else:
            return "Other"

    df["AllSymps_Impgrouped"] = df.apply(classify_group, axis=1)

    # Build comma-separated symptom list
    def get_all_symps(row):
        symps = []
        for col in sap_cols:
            if not pd.isna(row.get(col, np.nan)):
                symps.append(col.replace("SAP", "").replace("Total", "").strip())
        return ", ".join(sorted(set(s for s in symps if s)))

    df["AllImpSymptom"] = df.apply(get_all_symps, axis=1)

    return df


def add_adjunctive_metrics(
    df: pd.DataFrame,
    mnbi_df: Optional[pd.DataFrame] = None,
    pspw_df: Optional[pd.DataFrame] = None,
    id_col: str = "HospNum_Id",
) -> pd.DataFrame:
    """
    Merge MNBI and PSPW index dataframes into the impedance dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Impedance dataframe.
    mnbi_df : pd.DataFrame, optional
        MNBI dataframe with id_col and MNBI value columns.
    pspw_df : pd.DataFrame, optional
        PSPW dataframe with id_col and PSPW value columns.
    id_col : str
        Patient identifier column name.

    Returns
    -------
    pd.DataFrame
        Merged dataframe with adjunctive metrics added.
    """
    df = df.copy()

    if mnbi_df is not None and id_col in mnbi_df.columns:
        mnbi_df = mnbi_df.copy()
        df = df.merge(mnbi_df, on=id_col, how="left", suffixes=("", "_mnbi"))

    if pspw_df is not None and id_col in pspw_df.columns:
        pspw_df = pspw_df.copy()
        df = df.merge(pspw_df, on=id_col, how="left", suffixes=("", "_pspw"))

    return df


def gord_acid_imp_lyon(
    df: pd.DataFrame,
    aet_col: str = "MainAcidExpTotalClearanceChannelPercentTime",
    pathological_aet: float = 6.0,
    normal_aet: float = 4.0,
    mnbi_col: Optional[str] = None,
    mnbi_threshold: float = 2292.0,
    pspw_col: Optional[str] = None,
    pspw_threshold: float = 61.0,
    total_reflux_col: Optional[str] = None,
    total_reflux_threshold: float = 80.0,
) -> pd.DataFrame:
    """
    Lyon Consensus 2.0 classification for pH-impedance data.
    """
    df = df.copy()

    # Parse VisitDate if present
    if "VisitDate" in df.columns:
        s = (
            df["VisitDate"]
            .astype("string")
            .str.strip()
            .str.replace("_", "/", regex=False)
            .str.replace("-", "/", regex=False)
        )

        df["VisitDate"] = pd.to_datetime(
            s,
            format="%d/%m/%Y",
            errors="coerce"
        )

    # continue with the rest of the function...

    def classify_row(row):
        aet = pd.to_numeric(row.get(aet_col, np.nan), errors="coerce")

        if pd.isna(aet):
            return "Inconclusive", 0

        if aet > pathological_aet:
            return "Conclusive GORD", 1
        elif aet < normal_aet:
            return "GORD excluded", 0
        else:
            # Borderline — check adjunctive metrics
            adjunctive_positive = False

            if mnbi_col and mnbi_col in row.index:
                mnbi = pd.to_numeric(row.get(mnbi_col, np.nan), errors="coerce")
                if not pd.isna(mnbi) and mnbi < mnbi_threshold:
                    adjunctive_positive = True

            if pspw_col and pspw_col in row.index:
                pspw = pd.to_numeric(row.get(pspw_col, np.nan), errors="coerce")
                if not pd.isna(pspw) and pspw < pspw_threshold:
                    adjunctive_positive = True

            if total_reflux_col and total_reflux_col in row.index:
                total = pd.to_numeric(
                    row.get(total_reflux_col, np.nan), errors="coerce"
                )
                if not pd.isna(total) and total > total_reflux_threshold:
                    adjunctive_positive = True

            if adjunctive_positive:
                return "Conclusive GORD (adjunctive)", 1
            else:
                return "Inconclusive", 0

    results = df.apply(classify_row, axis=1)
    df["AcidReflux_Lyon"] = results.apply(lambda x: x[0])
    df["AcidReflux_Imp"] = results.apply(lambda x: x[1])

    return df


def gord_acid_imp(
    df: pd.DataFrame,
    aet_col: str = "MainAcidExpTotalClearanceChannelPercentTime",
    legacy_threshold: float = 4.2,
) -> pd.DataFrame:
    """
    Legacy 4.2% AET threshold classification (pre-Lyon).

    Kept for backward compatibility and historical comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Impedance dataframe.
    aet_col : str
        AET percentage column.
    legacy_threshold : float
        AET threshold (default 4.2%).

    Returns
    -------
    pd.DataFrame
        Dataframe with AcidReflux_Imp binary column added.
    """
    df = df.copy()
    aet = pd.to_numeric(df.get(aet_col, np.nan), errors="coerce")
    df["AcidReflux_Imp"] = (aet > legacy_threshold).astype(int)
    return df
