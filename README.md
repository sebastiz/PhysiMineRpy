# PhysiMiner — Python

Python port of [PhysiMineR](https://github.com/sebastiz/PhysiMineR).

Extraction, cleaning, and analysis of upper GI physiological data.

- **Acid reflux** classified per Lyon Consensus 2.0
- **Motility** diagnosed per Chicago Classification v4.0

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import physiminer as pm
import pandas as pd

# ── BRAVO ────────────────────────────────────────────────────────────────────
df = pm.data_bravo_clean(raw_bravo)
df = pm.data_bravo_day_labeller(df)
df = pm.data_bravo_symptoms(df)
df = pm.gord_acid_bravo_lyon(df)
df = pm.gord_bravo_wda_and_average(df)

# ── Impedance ────────────────────────────────────────────────────────────────
df = pm.data_imp_clean(raw_imp)
df = pm.data_imp_symptoms(df)
df = pm.gord_acid_imp_lyon(df, mnbi_col="M1_av", pspw_col="pspw_index")

# ── HRM ──────────────────────────────────────────────────────────────────────
df = pm.hrm_clean_up1(raw_hrm)
df = pm.hrm_diagnoses(df)                       # Chicago v4.0
df = pm.hrm_diagnoses(df, irp_uln=12)           # Diversatek system

# ── Merge ─────────────────────────────────────────────────────────────────────
combined = pm.triple_test_merge(hrm, imp, bravo, max_days_apart=365)

# ── Quality filter ────────────────────────────────────────────────────────────
clean = pm.filter_by_completeness(combined, col_complete_threshold=0.9, verbose=True)

# ── Symptoms ─────────────────────────────────────────────────────────────────
notes = pm.extract_symptoms(notes_df, text_cols="ClinicalNote")
notes = pm.symptom_burden_summary(notes)
```

## Function Reference

| Function | Description |
|---|---|
| `data_bravo_clean()` | Clean and deduplicate merged BRAVO export |
| `data_bravo_day_labeller()` | Standardise per-day AET column names to bravoDay1–bravoDay6 |
| `data_bravo_symptoms()` | Parse BRAVO symptom free text |
| `gord_acid_bravo_lyon()` | Lyon 2.0 three-tier BRAVO classification |
| `gord_bravo_wda_and_average()` | Worst-day and average-day AET analysis |
| `data_imp_clean()` | Clean merged pH-impedance export |
| `data_imp_symptoms()` | Categorise impedance symptoms by group |
| `add_adjunctive_metrics()` | Merge MNBI and PSPW index dataframes |
| `gord_acid_imp_lyon()` | Lyon 2.0 impedance classification with adjunctive support |
| `gord_acid_imp()` | Legacy 4.2% AET threshold classification |
| `hrm_clean_up1()` | Clean and collapse merged HRM export |
| `hrm_swallow_summary()` | Aggregate per-swallow metrics to study level |
| `hrm_diagnoses()` | Chicago Classification v4.0 motility diagnosis |
| `test_merge()` | Merge two test datasets by closest visit date |
| `triple_test_merge()` | Merge three test datasets by closest visit dates |
| `filter_by_completeness()` | Remove low-completeness columns/rows |
| `classify_symptom_association()` | Binary SAP/SI classification per Lyon 2.0 |
| `extract_symptoms()` | Extract symptom presence from free text |
| `symptom_burden_summary()` | Tally and categorise symptom burden |

## Lyon Consensus 2.0 Thresholds

| Metric | Threshold | Interpretation |
|---|---|---|
| AET (catheter, 24h) | > 6% | Conclusive GORD |
| AET (catheter, 24h) | 4–6% | Inconclusive |
| AET (catheter, 24h) | < 4% | GORD excluded |
| AET (BRAVO, 96h) | > 6% on ≥ 2 days | Conclusive GORD |
| AET (BRAVO, 96h) | < 4% all days | GORD excluded |
| MNBI | < 2292 Ω | Supportive (adjunctive) |
| PSPW index | < 61% | Supportive (adjunctive) |
| Total reflux episodes | > 80 | Supportive (adjunctive) |

## Chicago Classification v4.0 Diagnoses

| Diagnosis | IRP | Additional Criteria |
|---|---|---|
| Achalasia Type I | > ULN | 100% failed, no pan-oesophageal pressurisation |
| Achalasia Type II | > ULN | ≥ 20% pan-oesophageal pressurisation |
| Achalasia Type III | > ULN | ≥ 20% premature contractions |
| EGJ Outflow Obstruction | > ULN | Does not meet Types I–III |
| Distal Oesophageal Spasm | Normal | ≥ 20% premature contractions |
| Hypercontractile (Jackhammer) | Any | ≥ 20% swallows DCI > 8000 |
| Absent Contractility | Normal | 100% failed (DCI < 100) |
| IEM | Normal | ≥ 70% ineffective swallows |
| Fragmented Peristalsis | Normal | ≥ 50% fragmented AND mean DCI > 100 |
| Normal | Normal | None of the above |

IRP ULN: **15 mmHg** for Sierra/Medtronic, **12 mmHg** for Diversatek.
Set with `irp_uln` argument in `hrm_diagnoses()`.

## Requirements

- Python ≥ 3.9
- pandas ≥ 1.5
- numpy ≥ 1.23

## References

- Gyawali CP et al. (2021). Modern diagnosis of GERD: the Lyon Consensus. *Gut*, 70(7).
- Yadlapati R et al. (2021). Chicago Classification version 4.0. *Neurogastroenterology & Motility*, 33(1).
- Sweis R et al. (2014). PSPW index and nocturnal baseline impedance. *Neurogastroenterology & Motility*, 26(12).

## License

GPL-3 (matching original R package)
