# Diabetes Multi-Class Classification & Clustering

A complete end-to-end machine learning project for predicting diabetes status, Non-Diabetic (N), Pre-Diabetic (P), and Diabetic (Y), using clinical biomarker data.

## Project Structure

```
diabetes-prediction/
│
├── data/
│   └── raw/
│       └── dataset.csv               # Mendeley Diabetes Dataset
├── notebooks/
│   ├── eda_final.ipynb               # Exploratory Data Analysis
│   ├── modeling_classification.ipynb # Preprocessing and Classification Models
│   ├── modeling_clustering.ipynb     # Clustering Analysis
├── outputs/                          # Generated figures
└── README.md
```

## Dataset

| Property | Detail |
|---|---|
| **Source** | [Mendeley Data — wj9rwkp9c2/1](https://data.mendeley.com/datasets/wj9rwkp9c2/1) |
| **Rows** | 1,000 patients |
| **Features** | 11 (10 numerical, 1 categorical) |
| **Target** | `CLASS` → N (Non-Diabetic), P (Pre-Diabetic), Y (Diabetic) |
| **Missing Values** | None |
| **Class Distribution** | Y = 844 (84.4%), N = 103 (10.3%), P = 53 (5.3%) |
| **Class Imbalance** | Yes, handled with SMOTE inside cross-validation folds |

> Note: This repository includes data from the "Diabetes Dataset" by Ahlam Rashid, published on Mendeley Data (DOI: 10.17632/wj9rwkp9c2.1), licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

### Features

| Feature | Description | Clinical Range |
|---|---|---|
| `HbA1c` | Glycated Hemoglobin, primary diabetes test | <5.7% normal, ≥6.5% diabetic |
| `BMI` | Body Mass Index | 18.5–24.9 normal, ≥30 obese |
| `TG` | Triglycerides, insulin resistance marker | <150 mg/dL normal |
| `HDL` | Good cholesterol | >40 (M), >50 (F) mg/dL |
| `LDL` | Bad cholesterol | <100 mg/dL optimal |
| `Chol` | Total cholesterol | <200 mg/dL desirable |
| `VLDL` | Very-low-density lipoprotein | 2–30 mg/dL normal |
| `Urea` | Kidney function marker | 7–20 mg/dL normal |
| `Cr` | Creatinine, kidney filtration | 0.6–1.3 mg/dL normal |
| `AGE` | Patient age | 20–79 years |
| `Gender` | Biological sex | M / F |

## Quickstart

### 1. Clone & set up environment

```bash
git clone git@github.com:aakritip15/diabetes-prediction.git
cd diabetes-prediction
pip install -r requirements.txt
```

### 2. Download the dataset

Download from [Mendeley](https://data.mendeley.com/datasets/wj9rwkp9c2/1) and place it at:

```
data/raw/dataset.csv
```

### 3. Run notebooks in order

```bash
jupyter notebook
```

Open and run in this sequence:

```
1. eda_final.ipynb
2. modeling_classification.ipynb
3. modeling_clustering.ipynb
```

## Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
xgboost>=1.7.0
optuna>=3.0.0
scipy>=1.9.0
```

## Notebook Descriptions

### `eda_final.ipynb` : Exploratory Data Analysis

What it covers:
- Data inspection: shape, types, nulls, class distribution
- String cleaning: whitespace and casing inconsistencies in `CLASS` and `Gender`
- Duplicate visit detection: 39 patients with multiple recorded visits
- Outlier analysis via IQR: Cr (52 outliers, max=800), VLDL (74 outliers, max=35), Urea (65 outliers)
- Categorical univariate: Gender distribution and chi-square test (p=0.0001)
- Numerical univariate: histograms + KDE for all 10 features
- Bivariate: feature distributions split by N/P/Y class (boxplots)
- HbA1c deep dive: per-class statistics and WHO clinical threshold validation (91.7% match)
- BMI analysis: 100% of obese patients are diabetic in this dataset
- Multivariate: correlation heatmap + pairplot
- LDL/HDL ratio analysis: near-identical across classes (N=2.49, P=2.40, Y=2.56)

### `modeling_classification.ipynb` : Preprocessing & Classification

What it covers:
- String cleaning and ID column removal (`ID`, `No_Pation`)
- Outlier handling via percentile clipping (5th–95th)
- Stratified train/test split (80/20): Train=800, Test=200
- `ColumnTransformer` pipeline: `StandardScaler` for numerical, `OrdinalEncoder` for Gender
- SMOTE inside `ImbPipeline`, applied only within CV folds, no data leakage
- Four classifiers: Logistic Regression, Decision Tree, Random Forest, XGBoost
- Hyperparameter tuning with Optuna (50 trials, Bayesian search across RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, LogisticRegression)
- Error analysis: detailed inspection of misclassified samples

### `modeling_clustering.ipynb` : Clustering Analysis

What it covers:
- Feature scaling with `StandardScaler` before clustering
- PCA for dimensionality reduction: 2 PCs = 36.4%, 5 PCs = 72.9% variance explained
- K-Means with Elbow + Silhouette analysis (best k=3, silhouette=0.194)
- Agglomerative Clustering with dendrogram (Ward linkage)
- DBSCAN with k-distance plot for eps selection
- Cluster profiling: mean feature values per cluster mapped to clinical classes
- Full evaluation: Silhouette, Davies-Bouldin, ARI, NMI
- Algorithm comparison table across all three methods

Clustering on labeled data helps independently validate that the data has genuine separable structure.

## Results

### Classification

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **Random Forest** | **0.995** | **0.9913** | **0.9951** |
| XGBoost (baseline) | 0.995 | 0.9909 | 0.9949 |
| XGBoost (Optuna tuned) | 0.995 | 0.9909 | 0.9949 |
| Decision Tree | 0.980 | 0.9128 | 0.9776 |
| Logistic Regression | 0.955 | 0.8999 | 0.9574 |

Best Optuna CV Macro F1: **0.9686** (XGBClassifier, 50 trials)

Best Optuna hyperparameters: `n_estimators=277`, `learning_rate=0.1684`, `max_depth=4`, `min_child_weight=5`, `subsample=0.9136`, `colsample_bytree=0.9303`

### Clustering

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | ARI ↑ | NMI ↑ |
|---|---|---|---|---|
| **K-Means (k=3)** | **0.1939** | 1.6545 | **0.4801** | **0.3928** |
| Agglomerative (Ward) | 0.1868 | 1.6643 | 0.4419 | 0.3203 |
| DBSCAN | 0.0163 | 1.0578 | -0.0677 | 0.0832 |

DBSCAN found 9 clusters but labeled 59.7% of the data as noise, unsuitable for this dataset. Diabetes biomarkers form continuous gradients, not density-separated islands.

## Error Analysis

The best model (Random Forest) made **1 error out of 200 test samples** (0.5% error rate).

| True Class | Predicted | HbA1c | BMI | AGE | Notes |
|---|---|---|---|---|---|
| N | Y | 4.2 | 24.0 | 59 | HbA1c well within normal range; error likely driven by feature interactions |

**Key finding:** Zero critical errors - no diabetic patient (Y) was misclassified as non-diabetic (N). The clinically dangerous direction had zero errors.

The misclassified patient had HbA1c=4.2, which is at the dataset minimum and well below the mean HbA1c of correctly classified samples (8.22). The error was not due to a borderline value.

## Classification vs Clustering Comparison

| Approach | Best Method | Test Accuracy | ARI |
|---|---|---|---|
| Supervised (Classification) | XGBoost / Random Forest | 99.5% | ~0.99 |
| Unsupervised (Clustering) | K-Means (k=3) | - | 0.4801 |

Supervised classification dramatically outperforms unsupervised clustering, as expected - classifiers have access to labels during training. The value of clustering here is **independent validation**: the fact that K-Means (ARI=0.48) can partially recover the true labels without ever seeing them confirms the data has genuine, separable clinical structure.

## Note

> Extremely high scores (99%+) are probably inflated by patient data overlapping in the train and test sets. The dataset contains 39 patients with multiple visits. In production, a group-based split by patient ID would give a more honest evaluation of generalization to genuinely new patients.

## Future Enhancements

- **Group-based train/test split** by patient ID to prevent near-duplicate leakage and get honest performance estimates
- **Collect more pre-diabetic samples** - the P class has only 53 samples (5.3%), making it consistently the hardest class to predict and the most clinically important to improve
- **Semi-supervised learning** - use the discovered clustering structure (K-Means labels) to augment supervised training, especially for the underrepresented P class
- **SHAP explainability** - per-prediction feature importance for clinical interpretability and deployment trust
- **Cost-sensitive learning** - assign higher misclassification penalties to Y→N errors (missed diabetic), which are clinically the most dangerous
- **External validation** - evaluate on a held-out external cohort to test generalization beyond this specific dataset composition