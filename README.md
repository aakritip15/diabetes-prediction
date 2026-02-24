# Diabetes Multi-Class Classification & Clustering

A complete end-to-end machine learning project for predicting diabetes status — Non-Diabetic (N), Pre-Diabetic (P), and Diabetic (Y) — using clinical biomarker data.



##  Project Structure

```
diabetes-ml/
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


##  Dataset

| Property | Detail |
|---|---|
| **Source** | [Mendeley Data — wj9rwkp9c2/1](https://data.mendeley.com/datasets/wj9rwkp9c2/1) |
| **Rows** | ~1000 patients |
| **Features** | 11 (10 numerical, 1 categorical) |
| **Target** | `CLASS` → N (Non-Diabetic), P (Pre-Diabetic), Y (Diabetic) |
| **Missing Values** | None |
| **Class Imbalance** | Yes — Y dominates (~55%), handled with SMOTE |

Note: This repository includes data from the "Diabetes Dataset" by Ahlam Rashid,
published on Mendeley Data (DOI: 10.17632/wj9rwkp9c2.1), licensed under
Creative Commons Attribution 4.0 International (CC BY 4.0).

### Features

| Feature | Description | Clinical Range |
|---|---|---|
| `HbA1c` | Glycated Hemoglobin — primary diabetes test | <5.7% normal, ≥6.5% diabetic |
| `BMI` | Body Mass Index | 18.5–24.9 normal, ≥30 obese |
| `TG` | Triglycerides — insulin resistance marker | <150 normal |
| `HDL` | Good cholesterol | >40 (M), >50 (F) |
| `LDL` | Bad cholesterol | <100 optimal |
| `Chol` | Total cholesterol | <200 desirable |
| `VLDL` | Very-low-density lipoprotein | 2–30 normal |
| `Urea` | Kidney function marker | 7–20 normal |
| `Cr` | Creatinine — kidney filtration | 0.6–1.3 normal |
| `AGE` | Patient age | 20–79 years |
| `Gender` | Biological sex | M / F |



##  Quickstart

### 1. Clone & set up environment

```bash
git clone <your-repo-url>
cd diabetes-ml
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

### `eda_final.ipynb` — Exploratory Data Analysis
What it covers:
- Data inspection: shape, types, nulls, class distribution
- Categorical univariate: Gender distribution
- Numerical univariate: histograms + KDE for all 10 features
- Bivariate: feature distributions split by N/P/Y class (boxplots)
- Multivariate: correlation heatmap + pairplot



### `modeling_classification.ipynb` — Preprocessing & Classification
What it covers:
- String cleaning and ID column removal
- Outlier handling via percentile clipping (5th–95th)
- Stratified train/test split (80/20)
- ColumnTransformer pipeline (StandardScaler + OrdinalEncoder)
- SMOTE inside ImbPipeline (no data leakage)
- Four classifiers: Logistic Regression, Decision Tree, Random Forest, XGBoost
- Hyperparameter tuning with Optuna (80 trials, Bayesian search across 4 model types i.e RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, and LogisticRegression
 

### `modeling_clustering.ipynb` — Clustering Analysis
What it covers:
- PCA for dimensionality reduction and visualization
- K-Means with Elbow + Silhouette analysis to find optimal k
- Agglomerative Clustering with dendrogram (Ward linkage)
- DBSCAN with k-distance plot for eps selection
- Cluster profiling: mean feature values per cluster mapped to clinical classes
- Full evaluation: Silhouette, Davies-Bouldin, ARI, NMI

Clustering on labeled data helps in independent validation that the data has genuine separable structure.

