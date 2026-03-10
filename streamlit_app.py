import streamlit as st
import requests

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────────
st.title("📊 Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict churn risk.")
st.divider()

# ── Your AWS API URL ──────────────────────────────────────────────
API_URL = "http://18.202.54.130:8000/predict"  # ← replace with your IP

# ── Input Form ────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure)
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Partner", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    dependents = st.selectbox("Dependents", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    phone_service = st.selectbox("Phone Service", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    paperless_billing = st.selectbox("Paperless Billing", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

st.divider()

# ── Predict Button ────────────────────────────────────────────────
if st.button("🔮 Predict Churn", use_container_width=True):
    payload = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "PaperlessBilling": paperless_billing,
        "Contract": contract,
        "InternetService": internet_service,
        "MultipleLines": multiple_lines,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": "No",
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": "No",
        "PaymentMethod": payment_method
    }

    with st.spinner("Predicting..."):
        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()

            # ── Display Result ─────────────────────────────────────
            st.divider()
            churn_prob = result['churn_probability']
            risk = result['risk_level']
            churn = result['churn']

            if risk == "High":
                st.error(f"🔴 High Risk — {churn_prob*100:.1f}% chance of churning")
            elif risk == "Medium":
                st.warning(f"🟡 Medium Risk — {churn_prob*100:.1f}% chance of churning")
            else:
                st.success(f"🟢 Low Risk — {churn_prob*100:.1f}% chance of churning")

            # Progress bar
            st.progress(churn_prob)
            st.caption(f"Churn prediction: {'Yes' if churn else 'No'}")

        except Exception as e:
            st.error(f"API Error: {str(e)}")