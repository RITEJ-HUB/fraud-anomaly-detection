import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Fraud Anomaly Detection Dashboard", page_icon="🕵️", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "isolation_forest_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
SCORED_PATH = os.path.join(BASE_DIR, "scored_transactions.csv")
SAMPLE_PATH = os.path.join(BASE_DIR, "creditcard_sample.csv")

@st.cache_resource
def get_model_and_data():
    # If pre-trained model + scored data exist locally, use them (fast path for local dev)
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(SCORED_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        df = pd.read_csv(SCORED_PATH)
        return model, scaler, df

    # Otherwise (e.g. on Streamlit Cloud), train on the fly from the committed sample dataset
    df = pd.read_csv(SAMPLE_PATH)
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled'] = scaler.fit_transform(df[['Time']])

    feature_cols = [c for c in df.columns if c.startswith('V')] + ['Amount_scaled', 'Time_scaled']
    X = df[feature_cols]
    fraud_rate = df['Class'].mean()

    model = IsolationForest(n_estimators=200, contamination=fraud_rate, random_state=42, n_jobs=-1)
    model.fit(X)

    raw_preds = model.predict(X)
    df['predicted_anomaly'] = np.where(raw_preds == -1, 1, 0)
    df['anomaly_score'] = model.decision_function(X)

    return model, scaler, df

model, scaler, df = get_model_and_data()

st.title("🕵️ Fraud Anomaly Detection Dashboard")
st.caption("Unsupervised anomaly detection on credit card transactions using Isolation Forest")

# --- Sidebar controls ---
st.sidebar.header("Controls")
threshold_pct = st.sidebar.slider(
    "Anomaly sensitivity (% flagged as anomalies)",
    min_value=0.05, max_value=5.0, value=float(df['predicted_anomaly'].mean() * 100), step=0.05
)

cutoff = np.percentile(df['anomaly_score'], threshold_pct)
df['live_flag'] = (df['anomaly_score'] <= cutoff).astype(int)

# --- KPI row ---
total_txn = len(df)
flagged = int(df['live_flag'].sum())
actual_fraud = int(df['Class'].sum())
caught = int(((df['live_flag'] == 1) & (df['Class'] == 1)).sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{total_txn:,}")
col2.metric("Flagged as Anomalies", f"{flagged:,}")
col3.metric("Actual Frauds in Data", f"{actual_fraud:,}")
col4.metric("Frauds Caught", f"{caught} / {actual_fraud}")

st.divider()

# --- Charts ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("Anomaly Score Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[df['Class'] == 0]['anomaly_score'], bins=50, alpha=0.6, label='Normal', color='#4C72B0')
    ax.hist(df[df['Class'] == 1]['anomaly_score'], bins=50, alpha=0.6, label='Fraud', color='#C44E52')
    ax.axvline(cutoff, color='black', linestyle='--', label='Current threshold')
    ax.set_xlabel("Anomaly Score (lower = more suspicious)")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)

with c2:
    st.subheader("Flagged vs Actual Fraud")
    labels = ['True Positive\n(caught fraud)', 'False Positive\n(false alarm)', 'False Negative\n(missed fraud)']
    tp = caught
    fp = flagged - caught
    fn = actual_fraud - caught
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(labels, [tp, fp, fn], color=['#55A868', '#DD8452', '#C44E52'])
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

st.divider()

# --- Flagged transactions table ---
st.subheader("Flagged Transactions")
flagged_df = df[df['live_flag'] == 1].sort_values('anomaly_score')
display_cols = ['Time', 'Amount', 'Class', 'anomaly_score', 'live_flag']
st.dataframe(
    flagged_df[display_cols].rename(columns={
        'Class': 'Actually Fraud (1=yes)',
        'anomaly_score': 'Anomaly Score',
        'live_flag': 'Flagged'
    }),
    use_container_width=True,
    height=400
)

st.caption(f"Showing {len(flagged_df)} flagged transactions out of {total_txn:,} total, at the current sensitivity threshold.")
