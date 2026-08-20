# 🕵️ Fraud Anomaly Detection Dashboard

An unsupervised anomaly detection system for credit card fraud, built with Isolation Forest and visualized in an interactive Streamlit dashboard.

**🔗 Live demo: [fraud-anomaly-detection-fmhxnkkaads29n7rktdfbt.streamlit.app](https://fraud-anomaly-detection-fmhxnkkaads29n7rktdfbt.streamlit.app)**

## Overview

Fraud detection is a hard, real-world business problem: fraudulent transactions make up less than 0.2% of all activity, so traditional classification struggles without heavy labeling effort. This project takes an **unsupervised approach** — training an Isolation Forest to learn what "normal" transaction behavior looks like, then flagging anything that deviates significantly, without ever showing the model which transactions were actually fraud during training.

The dashboard lets a user interactively adjust detection sensitivity and see the tradeoff between catching more fraud and generating more false alarms — a real tradeoff fraud analysts face daily.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (Kaggle, ULB Machine Learning Group) — 284,807 anonymized European credit card transactions from September 2013, with 492 labeled frauds (0.17%). Features `V1`–`V28` are PCA-transformed for confidentiality; `Time` and `Amount` are the only raw features.

## Approach

1. **Preprocessing** — scale `Amount` and `Time` with `StandardScaler` to match the PCA-transformed feature range.
2. **Model** — train an `IsolationForest` (scikit-learn) on all transaction features, *excluding* the fraud label. `contamination` is set to match the real-world fraud rate (~0.17%).
3. **Scoring** — each transaction gets an anomaly score; the most negative scores are flagged as anomalies.
4. **Evaluation** — the true fraud labels are used only afterward, to measure how well the unsupervised model performed.
5. **Dashboard** — a Streamlit app lets a user adjust the anomaly threshold live and see flagged transactions, score distributions, and a true/false positive breakdown.

## Results (baseline model)

| Metric | Score |
|---|---|
| Precision | 0.26 |
| Recall | 0.26 |
| F1-score | 0.26 |
| Frauds caught | 126 / 492 |

These numbers reflect a genuine, honest baseline for unsupervised anomaly detection on this dataset — fraud detection without labels is a hard problem, and this establishes a starting point rather than a polished production model. Precision/recall are the metrics that matter here, not accuracy, since accuracy is misleading on such an imbalanced dataset (>99.8% of transactions are normal).

**Possible next steps to improve on this baseline:** ensemble multiple anomaly detectors (Isolation Forest + Local Outlier Factor + Autoencoder), engineer additional time-based or velocity features (e.g. transactions per hour per account), or move to a semi-supervised approach using the small number of known frauds.

## Tech Stack

- **Python** — Pandas, NumPy
- **scikit-learn** — Isolation Forest, StandardScaler, evaluation metrics
- **Streamlit** — interactive dashboard
- **Matplotlib** — visualizations
- **Joblib** — model persistence

## Project Structure

```
fraud-anomaly-detection/
├── fraud_detection_notebook.ipynb   # EDA, model training, evaluation
├── app.py                            # Streamlit dashboard
└── README.md
```

> Note: the raw dataset (`creditcard.csv`) and generated model/output files (`.pkl`, `scored_transactions.csv`) are not included in this repo due to file size. See setup instructions below to regenerate them.

## How to Run

1. Install dependencies:
   ```
   pip install pandas numpy scikit-learn matplotlib joblib streamlit
   ```
2. Download the [dataset from Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the project folder.
3. Run the notebook top to bottom to train the model and generate `isolation_forest_model.pkl`, `scaler.pkl`, and `scored_transactions.csv`:
   ```
   jupyter notebook fraud_detection_notebook.ipynb
   ```
4. Launch the dashboard:
   ```
   streamlit run app.py
   ```

## Author

Ritej — [GitHub](https://github.com/RITEJ-HUB)
