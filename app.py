import json
import os
from datetime import date, datetime, time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="FraudShield | Transaction Risk", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #0b1220 0%, #111c32 55%, #14263a 100%); color:#eef4ff;}
[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {padding: 1.5rem 1.8rem; border: 1px solid #293b59; border-radius: 22px;
background: linear-gradient(110deg, rgba(22,42,72,.95), rgba(17,28,49,.9)); margin-bottom:1.2rem;}
.hero h1 {margin:0; color:#f4f8ff; font-size:2.4rem;}
.hero p {color:#b7c7df; margin:.45rem 0 0 0;}
.panel {background:rgba(21,34,55,.82); border:1px solid #2a3b57; border-radius:18px; padding:1.2rem;}
.small {color:#aebed4; font-size:.92rem;}
div[data-testid="stMetric"] {background:rgba(26,43,68,.8); padding:14px; border:1px solid #2b3d5a; border-radius:14px;}
</style>
<div class="hero">
  <h1>🛡️ FraudShield</h1>
  <p>Credit-card transaction screening · Interactive machine-learning risk assessment</p>
</div>
""", unsafe_allow_html=True)

MODEL_PATH = os.getenv("MODEL_PATH", "fraud_model.joblib")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "threshold.json")
FEATURES = ["Amount", "MerchantID", "Hour", "DayOfWeek", "Month", "TransactionType", "Location"]

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def get_threshold():
    try:
        with open(THRESHOLD_PATH, encoding="utf-8") as f:
            return float(json.load(f).get("threshold", 0.5))
    except Exception:
        return 0.5

model = load_model()
threshold = get_threshold()

with st.sidebar:
    st.header("⚙️ Risk settings")
    st.caption("Adjust the decision threshold to control how cautious the detector is.")
    threshold_override = st.slider("Fraud threshold", 0.05, 0.95, float(threshold), 0.01)
    st.caption("Higher threshold usually means fewer alerts and can improve precision, but may miss more fraud.")
    st.divider()
    st.markdown("**How it works**")
    st.write("The model estimates fraud probability. The threshold converts that score into a review decision.")
    st.caption("This is a decision-support demo, not a guarantee or a substitute for fraud operations review.")

if model is None:
    st.error("Model file not found. Run `python train_model.py` first, or open `FraudShield_Colab.ipynb` in Google Colab and run all cells.")
    st.stop()

st.markdown("### Enter transaction details")
st.markdown('<p class="small">Fill in the fields below, then run the prediction. Use realistic values from your own transaction domain.</p>', unsafe_allow_html=True)

with st.form("transaction_form"):
    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input("Transaction amount", min_value=0.0, max_value=10000000.0, value=125.50, step=10.0, help="Amount in the currency used by your dataset.")
        merchant_id = st.number_input("Merchant ID", min_value=0, max_value=1000000000, value=1001, step=1)
        transaction_type = st.selectbox("Transaction type", ["Online", "POS", "ATM", "Transfer", "Other"])
    with c2:
        location = st.selectbox("Transaction location", ["Chicago", "Houston", "San Jose", "Dallas", "San Diego", "New York", "Los Angeles", "Other"])
        txn_date = st.date_input("Transaction date", value=date.today())
        txn_time = st.time_input("Transaction time", value=time(12, 0))
    submitted = st.form_submit_button("🔍 Analyze transaction", use_container_width=True, type="primary")

if submitted:
    dt = datetime.combine(txn_date, txn_time)
    sample = pd.DataFrame([{
        "Amount": float(amount),
        "MerchantID": int(merchant_id),
        "Hour": dt.hour,
        "DayOfWeek": dt.weekday(),
        "Month": dt.month,
        "TransactionType": transaction_type,
        "Location": location
    }])[FEATURES]
    probability = float(model.predict_proba(sample)[0, 1])
    is_fraud = probability >= threshold_override
    st.divider()
    st.markdown("### Assessment")
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated fraud probability", f"{probability:.1%}")
    m2.metric("Decision threshold", f"{threshold_override:.0%}")
    m3.metric("Model decision", "⚠️ Flag for review" if is_fraud else "✅ Likely legitimate")
    if is_fraud:
        st.error("Potentially fraudulent transaction — send for manual review or your established verification flow.")
    else:
        st.success("The transaction is below the current fraud threshold. This does not prove it is legitimate.")
    st.progress(min(max(probability, 0.0), 1.0), text="Estimated fraud risk")
    with st.expander("View prediction details"):
        st.json({
            "amount": float(amount), "merchant_id": int(merchant_id),
            "transaction_type": transaction_type, "location": location,
            "transaction_datetime": dt.isoformat(),
            "fraud_probability": round(probability, 6),
            "threshold": float(threshold_override),
            "decision": "flag_for_review" if is_fraud else "likely_legitimate"
        })

st.divider()
st.markdown("### Important notes")
st.markdown("""
- **Precision is not guaranteed.** Performance depends on how representative and correctly labelled your data is.
- The threshold is adjustable: raising it often reduces false alarms but can also miss fraud.
- Validate on a separate, recent dataset before using this for real payments. Avoid sending real card numbers or sensitive customer data to a demo app.
""")
