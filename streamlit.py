import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Diabetes ML Dashboard", layout="wide")
 
C_N, C_P, C_Y = "#16a34a", "#d97706", "#dc2626"
BLUE = "#2563eb"

# Data
class_counts = {"Y (Diabetic)": 844, "N (Non-Diabetic)": 103, "P (Pre-Diabetic)": 53}
num_cols = ["HbA1c", "BMI", "TG", "VLDL", "AGE", "Chol", "HDL", "LDL", "Urea", "Cr"]
means = {
    "N": [4.56, 22.37, 1.63, 0.94, 44.23, 4.27, 1.23, 2.63, 4.68, 62.80],
    "P": [6.00, 23.93, 2.13, 0.98, 43.28, 4.58, 1.13, 2.49, 4.51, 66.08],
    "Y": [8.88, 30.81, 2.45, 2.02, 55.31, 4.95, 1.21, 2.62, 5.22, 69.87],
}
gender_data = pd.DataFrame({
    "Class":  ["Non-Diabetic", "Pre-Diabetic", "Diabetic"] * 2,
    "Gender": ["Female"]*3 + ["Male"]*3,
    "Count":  [64, 17, 354, 39, 36, 490],
})
models = pd.DataFrame([
    {"Model": "Logistic Regression", "Accuracy": 0.955, "MacroF1": 0.8999, "N_F1": 0.89, "P_F1": 0.83, "Y_F1": 0.97},
    {"Model": "Decision Tree", "Accuracy": 0.990, "MacroF1": 0.9600, "N_F1": 1.00, "P_F1": 0.89, "Y_F1": 0.99},
    {"Model": "Decision Tree Tuned", "Accuracy": 0.980, "MacroF1": 0.9100, "N_F1": 1.00, "P_F1": 0.75, "Y_F1": 0.99},
    {"Model": "Random Forest", "Accuracy": 0.995, "MacroF1": 0.9913, "N_F1": 0.98, "P_F1": 1.00, "Y_F1": 1.00},
    {"Model": "XGBoost", "Accuracy": 0.995, "MacroF1": 0.9909, "N_F1": 0.98, "P_F1": 1.00, "Y_F1": 1.00},
    {"Model": "XGBoost (Optuna)", "Accuracy": 0.995, "MacroF1": 0.9909, "N_F1": 0.98, "P_F1": 1.00, "Y_F1": 1.00},
])
cm = np.array([[20, 0, 1], [0, 10, 0], [0, 0, 169]])
clustering = pd.DataFrame([
    {"Algorithm": "K-Means (k=3)", "Silhouette": 0.1939, "DB": 1.6545, "ARI": 0.4801, "NMI": 0.3928},
    {"Algorithm": "Agglomerative (Ward)", "Silhouette": 0.1868, "DB": 1.6643, "ARI": 0.4419, "NMI": 0.3203},
    {"Algorithm": "DBSCAN", "Silhouette": 0.0163, "DB": 1.0578, "ARI": -0.0677, "NMI": 0.0832},
])
cluster_profiles = {
    "Cluster 0": [5.56, 23.47, 1.92, 0.91, 43.59, 4.81, 1.18, 2.74, 4.44, 60.76],
    "Cluster 1": [8.67, 32.53, 2.21, 7.98, 48.03, 4.91, 1.08, 2.83, 4.86, 69.93],
    "Cluster 2": [9.07, 31.23, 2.42, 1.11, 57.04, 4.85, 1.14, 2.49, 5.04, 62.65],
}
pca_pcs, pca_var = ["2 PCs", "3 PCs", "4 PCs", "5 PCs"], [36.4, 51.7, 62.6, 72.9]
optuna_params = {"classifier": "XGBClassifier", "n_estimators": 277, "learning_rate": 0.1684, "max_depth": 4, "subsample": 0.9136}

# Header
st.title("Diabetes ML Insights")
st.markdown("### Analysis Report for 1,000 Patient Records")
st.divider()

# Siderbar
tab = st.sidebar.radio(" ", ["EDA", "Classification", "Clustering", "Optimization"])

if tab == "EDA":
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", "1,000")
    c2.metric("Diabetic (Y)", "844", delta="84.4%", delta_color="inverse")
    c3.metric("Non-Diabetic (N)", "103", delta="10.3%")
    c4.metric("Pre-Diabetic (P)", "53", delta="5.3%")

    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = go.Figure(go.Bar(x=list(class_counts.keys()), y=list(class_counts.values()), marker_color=[C_Y, C_N, C_P]))
        fig1.update_layout(title="Class Distribution", template="plotly_white", height=350)
        st.plotly_chart(fig1, width='stretch')
    
    with col_b:
        fig_gen = go.Figure()
        for g, color in [("Female", "#7c3aed"), ("Male", BLUE)]:
            df_g = gender_data[gender_data["Gender"] == g]
            fig_gen.add_trace(go.Bar(name=g, x=df_g["Class"], y=df_g["Count"], marker_color=color))
        fig_gen.update_layout(title="Gender Distribution by Class", barmode="group", template="plotly_white", height=350)
        st.plotly_chart(fig_gen, width='stretch')

    st.subheader("Biological Feature Analysis")
    fig2 = go.Figure()
    for cls, color, name in [("N", C_N, "Non-Diabetic"), ("P", C_P, "Pre-Diabetic"), ("Y", C_Y, "Diabetic")]:
        fig2.add_trace(go.Bar(name=name, x=num_cols, y=means[cls], marker_color=color))
    fig2.update_layout(title="Mean Feature Values per Class", barmode="group", template="plotly_white")
    st.plotly_chart(fig2, width='stretch')

elif tab == "Classification":
    c1, c2, c3 = st.columns(3)
    c1.metric("Top Accuracy", "99.5%", "RF / XGB")
    c2.metric("Macro F1-Score", "0.9913", "Random Forest")
    c3.metric("Test Error Rate", "0.5%", "-1 Sample")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig_f1 = go.Figure()
        for col, color, name in [("N_F1", C_N, "N"), ("P_F1", C_P, "P"), ("Y_F1", C_Y, "Y")]:
            fig_f1.add_trace(go.Bar(name=f"F1: {name}", x=models["Model"], y=models[col], marker_color=color))
        fig_f1.update_layout(title="Per-Class F1 Score Comparison", barmode="group", template="plotly_white", yaxis_range=[0.6, 1.1])
        st.plotly_chart(fig_f1, width='stretch')
    
    with col_b:
        st.write("#### Error Analysis")
        st.info("**Sample Misclassification:**\n- True: **N** | Pred: **Y**\n- Patient 59 yrs | HbA1c 4.2\n- Risk: High (False Positive)")
        fig_cm = go.Figure(go.Heatmap(z=cm, x=["Pred N", "Pred P", "Pred Y"], y=["True N", "True P", "True Y"], colorscale='Greens', text=cm, texttemplate="<b>%{text}</b>"))
        fig_cm.update_layout(title="Confusion Matrix (RF)", height=300, template="plotly_white")
        st.plotly_chart(fig_cm, width='stretch')

elif tab == "Clustering":
    st.subheader("Unsupervised Learning: K-Means (k=3)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Silhouette Score", "0.1939")
    c2.metric("ARI", "0.4801")
    c3.metric("DB Index", "1.6545")

    fig_prof = go.Figure()
    for (name, vals), color in zip(cluster_profiles.items(), [BLUE, C_Y, C_P]):
        fig_prof.add_trace(go.Scatter(x=num_cols, y=vals, mode="lines+markers", name=name, line=dict(color=color)))
    fig_prof.update_layout(title="Cluster Centroid Feature Profiles", template="plotly_white")
    st.plotly_chart(fig_prof, width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        fig_pca = go.Figure(go.Bar(x=pca_pcs, y=pca_var, marker_color=BLUE))
        fig_pca.update_layout(title="PCA Cumulative Variance", template="plotly_white", yaxis_title="% Variance")
        st.plotly_chart(fig_pca, width='stretch')
    with col_b:
        st.write("#### Clustering Observations")
        st.success("- **Cluster 1** perfectly identifies high VLDL/BMI patterns.\n- **DBSCAN** yielded 59.7% noise, suggesting biomarker continuity.")

elif tab == "Optimization":
    st.subheader("Optuna Hyperparameter Search")
    df_params = pd.DataFrame([optuna_params]).T.rename(columns={0: "Value"})
    df_params["Value"] = df_params["Value"].astype(str)
    st.table(df_params)
    st.info("The optimization improved the Macro F1 of XGBoost to 0.9909 over 50 trials.")
    