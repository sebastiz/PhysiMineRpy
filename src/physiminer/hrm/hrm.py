"""
PhysiMiner - High-Resolution Manometry (HRM) Module
Cleaning, swallow aggregation, and Chicago Classification v4.0 diagnosis.
"""

import pandas as pd
import numpy as np
import re
from typing import Optional


def hrm_clean_up1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalise column names
    df.columns = df.columns.astype(str).str.strip()

    # Drop raw per-swallow numeric trace columns
    trace_cols = [c for c in df.columns if re.match(r"Num\d+", str(c))]
    df = df.drop(columns=trace_cols, errors="ignore")

    # Columns that should never be converted to numeric
    protected_text_cols = {
        "chicagoclassification",
        "procedure",
        "physician",
        "referringphysician",
        "gender",
        "operator",
        "hospnum_id",
        "hrm_id",
        "dobage",
        "height",
    }

    # Parse date columns only
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(
                df[col],
                errors="coerce",
                dayfirst=True
            )

    # Convert only mostly numeric columns
    for col in df.columns:
        col_key = col.lower().strip()

        if col_key in protected_text_cols:
            continue

        if "date" in col_key:
            continue

        if df[col].dtype == "object" or str(df[col].dtype) == "string":
            non_missing = df[col].dropna()

            if len(non_missing) == 0:
                continue

            converted = pd.to_numeric(non_missing, errors="coerce")
            proportion_numeric = converted.notna().mean()

            if proportion_numeric >= 0.8:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Deduplicate to one row per HRM study ID
    if "HRM_Id" in df.columns:
        id_col = "HRM_Id"

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        agg_dict = {}

        for col in df.columns:
            if col == id_col:
                continue

            if col in numeric_cols:
                agg_dict[col] = "mean"
            else:
                agg_dict[col] = "first"

        df = df.groupby(id_col, as_index=False).agg(agg_dict)

    return df.reset_index(drop=True)


def hrm_swallow_summary(
    df: pd.DataFrame,
    id_col: str = "HRM_Id",
    dci_col: str = "DCImmHgcms",
    dl_col: str = "DistallatencyS",
    cfv_col: str = "ContractilefrontvelocityCms",
    break_col: str = "BreakCm",
    large_break_threshold: float = 5.0,
) -> pd.DataFrame:
    """
    Aggregate per-swallow metrics to study level.

    Parameters
    ----------
    df : pd.DataFrame
        Per-swallow dataframe.
    id_col : str
        Study identifier column.
    dci_col : str
        Distal contractile integral column (mmHg·cm·s).
    dl_col : str
        Distal latency column (s).
    cfv_col : str
        Contractile front velocity column (cm/s).
    break_col : str
        Peristaltic break size column (cm).
    large_break_threshold : float
        Minimum break size considered 'large' (default 5 cm).

    Returns
    -------
    pd.DataFrame
        Study-level summary with one row per HRM_Id.
    """
    df = df.copy()

    for col in [dci_col, dl_col, cfv_col, break_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    summaries = []
    for study_id, group in df.groupby(id_col):
        row = {id_col: study_id}

        if dci_col in group.columns:
            dci = group[dci_col].dropna()
            row["DCI_mean"] = dci.mean()
            row["DCI_median"] = dci.median()
            n = len(dci)
            row["pct_failed"] = (dci < 100).sum() / n * 100 if n > 0 else np.nan
            row["pct_weak"] = ((dci >= 100) & (dci < 450)).sum() / n * 100 if n > 0 else np.nan
            row["pct_hypercontract"] = (dci > 8000).sum() / n * 100 if n > 0 else np.nan

        if dl_col in group.columns:
            dl = group[dl_col].dropna()
            row["DL_mean"] = dl.mean()
            n = len(dl)
            row["pct_premature"] = (dl < 4.5).sum() / n * 100 if n > 0 else np.nan

        if cfv_col in group.columns:
            row["CFV_mean"] = group[cfv_col].dropna().mean()

        if break_col in group.columns:
            breaks = group[break_col].dropna()
            n = len(breaks)
            row["pct_large_break"] = (breaks >= large_break_threshold).sum() / n * 100 if n > 0 else np.nan
            row["pct_small_break"] = (
                (breaks > 0) & (breaks < large_break_threshold)
            ).sum() / n * 100 if n > 0 else np.nan

        summaries.append(row)

    return pd.DataFrame(summaries)


def hrm_diagnoses(
    df: pd.DataFrame,
    irp_col: str = "ResidualmeanmmHg",
    irp_uln: float = 15.0,
    failed_col: str = "failedChicagoClassification",
    paneso_col: str = "panesophagealpressurization",
    premature_col: str = "prematurecontraction",
    rapid_col: str = "rapidcontraction",
    dci_col: str = "DistalcontractileintegralmeanmmHgcms",
    large_break_col: str = "largebreaks",
    small_break_col: str = "smallbreaks",
    simultaneous_col: str = "Simultaneous",
) -> pd.DataFrame:
    """
    Apply Chicago Classification v4.0 motility diagnosis.

    Diagnosis hierarchy (checked in order):
    1. Achalasia Type I   — IRP > ULN, 100% failed, no pan-oesophageal pressurisation
    2. Achalasia Type II  — IRP > ULN, ≥ 20% pan-oesophageal pressurisation
    3. Achalasia Type III — IRP > ULN, ≥ 20% premature contractions
    4. EGJ Outflow Obstruction — IRP > ULN, not Types I–III
    5. Distal Oesophageal Spasm — normal IRP, ≥ 20% premature contractions
    6. Hypercontractile Oesophagus — ≥ 20% swallows DCI > 8000
    7. Absent Contractility — normal IRP, 100% failed (DCI < 100)
    8. IEM — normal IRP, ≥ 70% ineffective swallows
    9. Fragmented Peristalsis — normal IRP, ≥ 50% large break AND mean DCI > 100
    10. Normal

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned HRM dataframe.
    irp_col : str
        Integrated Relaxation Pressure (mean) column.
    irp_uln : float
        Upper limit of normal for IRP (15 mmHg Sierra, 12 mmHg Diversatek).
    failed_col : str
        Percentage of failed peristalsis column.
    paneso_col : str
        Percentage of pan-oesophageal pressurisation column.
    premature_col : str
        Percentage of premature contractions column.
    rapid_col : str
        Percentage of rapid contractions column.
    dci_col : str
        Mean DCI column (mmHg·cm·s).
    large_break_col : str
        Percentage of swallows with large peristaltic break (≥ 5 cm).
    small_break_col : str
        Percentage of swallows with small peristaltic break (< 5 cm).
    simultaneous_col : str
        Percentage of simultaneous contractions column.

    Returns
    -------
    pd.DataFrame
        Dataframe with ChicagoV4Diagnosis and ChicagoV4DiagnosisGroup added.
    """
    df = df.copy()

    # Coerce all relevant columns
    for col in [
        irp_col, failed_col, paneso_col, premature_col, rapid_col,
        dci_col, large_break_col, small_break_col, simultaneous_col,
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def get(row, col, default=0.0):
        if col not in row.index:
            return default
        v = row[col]
        return float(v) if not pd.isna(v) else default

    def diagnose(row):
        irp = get(row, irp_col, 0.0)
        failed = get(row, failed_col, 0.0)
        paneso = get(row, paneso_col, 0.0)
        premature = get(row, premature_col, 0.0)
        rapid = get(row, rapid_col, 0.0)
        dci = get(row, dci_col, 0.0)
        large_break = get(row, large_break_col, 0.0)
        small_break = get(row, small_break_col, 0.0)

        elevated_irp = irp > irp_uln

        # ── Achalasia types (elevated IRP required) ──────────────────────────
        if elevated_irp:
            if failed >= 100 and paneso < 20:
                return "Achalasia Type I", "Achalasia"
            if paneso >= 20:
                return "Achalasia Type II", "Achalasia"
            if premature >= 20:
                return "Achalasia Type III", "Achalasia"
            return "EGJ Outflow Obstruction", "EGJ Outflow Obstruction"

        # ── Normal IRP diagnoses ─────────────────────────────────────────────
        # Hypercontractile oesophagus (can have any IRP)
        if rapid >= 20 and dci > 8000:
            return "Hypercontractile Oesophagus (Jackhammer)", "Major Peristalsis Disorder"

        # Distal oesophageal spasm
        if premature >= 20:
            return "Distal Oesophageal Spasm", "Major Peristalsis Disorder"

        # Absent contractility
        if failed >= 100:
            return "Absent Contractility", "Major Peristalsis Disorder"

        # IEM — ≥ 70% ineffective (failed + weak)
        if failed >= 70:
            return "Ineffective Oesophageal Motility", "Minor Peristalsis Disorder"

        # Fragmented peristalsis — ≥ 50% fragmented (any break size) AND mean DCI > 100
        if (large_break >= 50 or small_break >= 50) and dci > 100:
            return "Fragmented Peristalsis", "Minor Peristalsis Disorder"

        return "Normal", "Normal"

    results = df.apply(diagnose, axis=1)
    df["ChicagoV4Diagnosis"] = results.apply(lambda x: x[0])
    df["ChicagoV4DiagnosisGroup"] = results.apply(lambda x: x[1])

    return df
