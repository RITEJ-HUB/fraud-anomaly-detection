import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Anomaly Detection Dashboard", page_icon="🕵️", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "scored_transactions.csv"))
    return df

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "isolation_forest_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    return model, scaler

df = load_data()
model, scaler = load_model()

st.title("🕵️ Fraud Anomaly Detection Dashboard")
st.caption("Unsupervised anomaly detection on credit card transactions using Isolation Forest")

# --- Sidebar controls ---
st.sidebar.header("Controls")
threshold_pct = st.sidebar.slider(
    "Anomaly sensitivity (% flagged as anomalies)",
    min_value=0.05, max_value=5.0, value=float(df['predicted_anomaly'].mean() * 100), step=0.05
)

# Recompute flags live based on anomaly_score using the chosen percentile threshold
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
