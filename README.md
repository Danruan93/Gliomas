# Gliomas Radiomic Dataset

**Overview**

This dataset contains radiomic features and corresponding clinical & pathological information for glioma, including multi-modal MRI radiomic features, clinical data, pathological information, and lesion contrast enhancement annotation files.

**MRI Modalities**

Radiomic features are extracted from four conventional MRI sequences:
T1WI (t1)
T2WI (t2)
FLAIR (flair)
Contrast-enhanced T1WI (t1c / T1CE)

**Dataset Split**

FZ: Training cohort 
BJ: External test cohort 1 
XM: External test cohort 2 

**File Description**

All files are grouped by cohort (FZ / BJ / XM) with consistent naming rules.
1. Radiomic feature files (.csv)
[cohort]_flair_features.csv: FLAIR sequence radiomic features
[cohort]_t1_features.csv: T1WI sequence radiomic features
[cohort]_t1c_features.csv: Contrast-enhanced T1WI (T1CE) radiomic features
[cohort]_t2_features.csv: T2WI sequence radiomic features
2. Clinical & pathological & annotation files (.xlsx/.csv)
[cohort]_clinical_information.csv: Clinical information of patients
[cohort]_Pathology.xlsx: Pathological diagnosis & related parameters
[cohort]_Lesion_contrast_enhancement.xlsx: Lesion contrast enhancement annotation

**Usage Notes**
FZ is used for model training, validation, and feature selection.
BJ and XM are used as independent external test sets for model generalization evaluation.
All radiomic features are extracted in consistent processing pipeline.
