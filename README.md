# Bank Transaction Data Analysis: End-to-End Customer Segmentation and Classification

An end-to-end machine learning project utilizing customer banking transaction data. The project implements an unsupervised clustering pipeline to discover behavioral customer segments, followed by a supervised classification pipeline to accurately predict customer segments for downstream banking operations.

## Table of Contents

- Project Overview
- Business Problem and Motivation
- Dataset Description
- Project Pipeline
- Data Preprocessing and Feature Engineering
- Phase 1: Unsupervised Learning (Clustering)
  - Finding Optimal Clusters (Elbow Method)
  - Model Training and Dimensionality Reduction (PCA)
  - Cluster Evaluation and Interpretation
- Phase 2: Supervised Learning (Classification)
  - Model Exploration (Decision Tree vs Random Forest)
  - Hyperparameter Tuning (GridSearchCV)
  - Model Evaluation
- Repository Structure
- Saved Model Artifacts
- Requirements and Setup
- Usage Guide
- Author

---

## Project Overview

Modern financial institutions process millions of transactions daily across diverse channels (ATM, Branch, Online). Understanding transaction behavior and demographic patterns enables banks to:

1. Segment customers into actionable groups for targeted marketing and relationship management.
2. Identify anomalous usage patterns or channel preferences.
3. Automatically classify incoming customer transactions into behavioral profiles.

This repository demonstrates the full lifecycle: from exploratory data analysis and outlier handling to K-Means clustering, PCA dimensionality reduction, and multi-class classification modeling with hyperparameter tuning.

---

## Business Problem and Motivation

Customer heterogeneity makes a one-size-fits-all banking strategy ineffective. Financial services need data-driven approaches to personalize service offerings, optimize channel allocation, and improve customer retention.

- Unsupervised Objective: Identify distinct customer behavioral patterns based on transaction amounts, account balance, transaction duration, customer age, and interaction channels without prior labels.
- Supervised Objective: Train a robust classifier to predict customer segment labels (`Target`) so that new transactions can be categorized in real time without having to re-cluster the entire dataset.

---

## Dataset Description

The original dataset (`bank_transactions_data_edited.csv`) consists of 2,537 rows and 16 features capturing customer demographics, transaction details, and operational metadata.

### Feature Summary

| Feature Name | Type | Description |
|---|---|---|
| TransactionID | Categorical | Unique transaction identifier (dropped during preprocessing) |
| AccountID | Categorical | Unique customer account identifier (dropped during preprocessing) |
| TransactionAmount | Numerical | Monetary value of the transaction |
| TransactionDate | Datetime | Timestamp of the transaction |
| PreviousTransactionDate | Datetime | Timestamp of the previous transaction |
| TransactionType | Categorical | Type of transaction (e.g., Debit, Credit) |
| Location | Categorical | City or location where transaction took place |
| Channel | Categorical | Access channel used (ATM, Branch, Online) |
| CustomerAge | Numerical | Age of the customer |
| CustomerOccupation | Categorical | Occupation of the account holder (Doctor, Student, etc.) |
| TransactionDuration | Numerical | Duration of transaction in seconds |
| LoginAttempts | Numerical | Number of login attempts recorded |
| AccountBalance | Numerical | Remaining balance in the account |
| DeviceID | Categorical | Device identifier used (dropped during preprocessing) |
| IP Address | Categorical | IP address of transaction (dropped during preprocessing) |
| MerchantID | Categorical | Identifier of the merchant (dropped during preprocessing) |

---

## Project Pipeline

1. Data Exploration and Quality Audit: Inspection of column types, distributions, correlations, and missing values.
2. Data Cleansing and Outlier Treatment: Removal of missing rows, duplicate elimination, non-informative identifier filtering, and IQR-based outlier trimming.
3. Feature Engineering: Discretization/binning of `CustomerAge` into demographic groups (`Muda`, `Dewasa`, `Senior`) and categorical label encoding.
4. Feature Normalization: Standardization of continuous numerical features using `StandardScaler`.
5. Clustering Pipeline: Determination of optimal cluster count via `KElbowVisualizer`, K-Means fitting (k=4), Silhouette Score evaluation, and PCA 2D projection.
6. Cluster Profiling: Inverse transformation of cluster features to extract real-world statistical interpretations per segment.
7. Classification Pipeline: Stratified 80:20 train-test split, baseline training with Decision Tree, ensemble training with Random Forest, and hyperparameter optimization using `GridSearchCV`.
8. Artifact Serialization: Export of trained models (`.h5`) and processed datasets (`.csv`).

---

## Data Preprocessing and Feature Engineering

### 1. Data Cleaning
- Missing Values and Duplicates: Dropped missing entries and duplicate rows to prevent distortion.
- Identifier Dropping: Removed non-predictive columns (`TransactionID`, `AccountID`, `DeviceID`, `IP Address`, `MerchantID`, `TransactionDate`, `PreviousTransactionDate`).

### 2. Outlier Handling
- Outlier filtering was applied across numerical features using the Interquartile Range (IQR) rule:
  `[Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]`
- Sample count before outlier removal: 2,135 rows.
- Sample count after outlier removal: 1,945 rows.

### 3. Feature Engineering and Encoding
- Categorical Encoding: `TransactionType`, `Location`, `Channel`, and `CustomerOccupation` were converted to numerical representations using `LabelEncoder`.
- Age Discretization: `CustomerAge` was binned into three distinct categories:
  - Muda: age <= 30
  - Dewasa: 31 <= age <= 50
  - Senior: age > 50
- Scaling: Numerical features (`TransactionAmount`, `CustomerAge`, `TransactionDuration`, `LoginAttempts`, `AccountBalance`) were standardized using `StandardScaler` (zero mean, unit variance).

---

## Phase 1: Unsupervised Learning (Clustering)

- Notebook: `[Clustering]_Submission_Akhir_BMLP_Muhamad_Nabhan_Fadhlurrohman.ipynb`

### Finding Optimal Clusters
The optimal number of clusters was determined using the Elbow Method with `yellowbrick.cluster.KElbowVisualizer` evaluating k from 2 through 10. The elbow point was identified at **k = 4**.

### Model Training
- Algorithm: `KMeans(n_clusters=4, random_state=42, n_init=10)`
- Total segmented samples: 1,945 rows

### Evaluation Metrics
- Silhouette Score: **0.4447**, indicating good cluster cohesion and separation.
- Dimensionality Reduction: Principal Component Analysis (PCA) was fitted to reduce input dimensions to 2 principal components for cluster boundary visualization.
  - PC1 explained variance: ~95.74%
  - PC2 explained variance: ~1.38%

### Cluster Profiles and Interpretation
Using `inverse_transform` to examine cluster characteristics in their natural units:
- Cluster 0: Moderate transaction amounts, balanced channel usage across branches and online, medium-to-high account balances.
- Cluster 1: High frequency of ATM transactions, younger demographic skew, smaller transaction amounts with shorter durations.
- Cluster 2: Senior demographic dominance, higher average account balances, preference for branch/ATM channels.
- Cluster 3: High transaction amounts with higher durations, active business/working segment.

### Exported Datasets
- `data_clustering.csv`: Scaled features paired with the assigned cluster label (`Target`).
- `data_clustering_inverse.csv`: Unscaled original values paired with the assigned cluster label (`Target`).

---

## Phase 2: Supervised Learning (Classification)

- Notebook: `[Klasifikasi]_Submission_Akhir_BMLP_Muhamad_Nabhan_Fadhlurrohman.ipynb`

The goal of this phase is to construct a predictive model that maps customer transaction characteristics directly to their cluster target (`Target`), enabling instant classification of prospective transactions.

### Data Partitioning
- Source data: `data_clustering_inverse.csv`
- Split ratio: 80% training (1,556 samples), 20% testing (389 samples)
- Stratification: Stratified by `Target` to preserve class distributions (`random_state=42`).

### Models Trained and Evaluated

1. Baseline: `DecisionTreeClassifier(random_state=42)`
2. Comparison: `RandomForestClassifier(random_state=42)`
3. Hyperparameter Tuned: `GridSearchCV` on Random Forest
   - Parameter grid searched:
     - `n_estimators`: [50, 100, 200]
     - `max_depth`: [5, 10, 20, None]
     - `min_samples_split`: [2, 5, 10]
   - Cross-Validation: 3-fold CV with `scoring='f1_weighted'`
   - Optimal parameters: `{'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 50}`

### Performance Comparison

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
|---|---|---|---|---|
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest (Tuned) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Because the target labels were derived from deterministic cluster boundaries over the same feature space, tree-based models were able to separate the classes on the test set. The tuned Random Forest model provides strong generalization through constrained tree depth (`max_depth=5`) and ensemble averaging.

---

## Repository Structure

```
Bank-Transaction-Data/
|-- [Clustering]_Submission_Akhir_BMLP_Muhamad_Nabhan_Fadhlurrohman.ipynb
|-- [Klasifikasi]_Submission_Akhir_BMLP_Muhamad_Nabhan_Fadhlurrohman.ipynb
|-- data_clustering.csv
|-- data_clustering_inverse.csv
|-- decision_tree_model.h5
|-- explore_RandomForest_classification.h5
|-- model_clustering.h5
|-- PCA_model_clustering.h5
|-- tuning_classification.h5
|-- .gitignore
|-- README.md
```

---

## Saved Model Artifacts

All models are serialized using `joblib` in `.h5` format for deployment and reproducibility:

- `model_clustering.h5`: Trained K-Means clustering model (k=4).
- `PCA_model_clustering.h5`: Fitted 2-component PCA model for dimensionality reduction.
- `decision_tree_model.h5`: Baseline Decision Tree classification model.
- `explore_RandomForest_classification.h5`: Default Random Forest classification model.
- `tuning_classification.h5`: Hyperparameter-tuned Random Forest model (`max_depth=5`, `n_estimators=50`).

---

## Requirements and Setup

### Prerequisites
- Python 3.8 or later
- Jupyter Notebook or JupyterLab

### Python Dependencies
Install required packages using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn yellowbrick joblib
```

---

## Usage Guide

### 1. Running the Clustering Notebook
1. Open `[Clustering]_Submission_Akhir_BMLP_Muhamad_Nabhan_Fadhlurrohman.ipynb`.
2. Execute the cells sequentially to run EDA, preprocessing, Elbow Method optimization, K-Means clustering, and PCA visualization.
3. The notebook outputs `data_clustering.csv`, `data_clustering_inverse.csv`, `model_clustering.h5`, and `PCA_model_clustering.h5`.

### 2. Running the Classification Notebook
1. Open `[Klasifikasi]_Submission_Akhir_BMLP_Muhamad_Nabhan_Fadhlurrohman.ipynb`.
2. Ensure `data_clustering_inverse.csv` is present in the working directory.
3. Execute all cells to perform data splitting, train Decision Tree and Random Forest models, run `GridSearchCV`, and export `tuning_classification.h5`.

### 3. Loading Saved Models for Inference

```python
import joblib
import pandas as pd

# Load tuned classification model
classifier = joblib.load('tuning_classification.h5')

# Load clustering model
kmeans = joblib.load('model_clustering.h5')

# Example prediction
# sample_data should contain the encoded feature columns matching training
# prediction = classifier.predict(sample_data)
# print("Predicted Segment:", prediction)
```

---

## Author

- Name: Muhamad Nabhan Fadhlurrohman
- Project: Belajar Machine Learning Terapan / Pemula (BMLP) Submission
- GitHub: [diabloirn](https://github.com/diabloirn)
