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
1. Dataset Usage

  FZ cohort serves as the training dataset, which is used for model training, internal validation, and radiomic feature selection.
  BJ and XM cohorts are utilized as independent external test datasets to evaluate the generalization performance of the developed models.
  
2. Script Functions

  train_data.r: Processes the training dataset (FZ cohort), performs dimensionality reduction, and selects key radiomic features.
  test1.r & test2.r: Conduct preprocessing and feature alignment for the BJ and XM test datasets, respectively.
  models.r: Implements the construction of predictive models (e.g., classification/regression models) and computes the corresponding prediction results (e.g., performance metrics, prediction outputs).
  
3. Feature Extraction

  All radiomic features are extracted using a standardized and consistent processing pipeline to ensure reproducibility.
