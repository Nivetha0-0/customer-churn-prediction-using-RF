import streamlit as st
import pickle
import pandas as pd

# ─── Load saved objects ────────────────────────────────────────────────
with open('rf_churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

# ─── App layout ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("Telco Customer Churn Predictor")
st.write("Enter customer details to predict if they will **stay** or **churn** (leave).")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.radio("Senior Citizen", ["No", "Yes"], horizontal=True)
        partner = st.radio("Partner", ["No", "Yes"], horizontal=True)
        dependents = st.radio("Dependents", ["No", "Yes"], horizontal=True)
        tenure = st.slider("Tenure (months)", 0, 80, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 10.0, 150.0, 70.0)

    with col2:
        phone_service = st.radio("Phone Service", ["No", "Yes"], horizontal=True)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.radio("Paperless Billing", ["No", "Yes"], horizontal=True)
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])

    predict_button = st.form_submit_button("Predict", type="primary")

# ─── Make prediction when button is clicked ────────────────────────────
if predict_button:
    # Create input DataFrame (match your training columns as closely as possible)
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior == "Yes" else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': ['No phone service' if phone_service == "No" else 'No'],
        'InternetService': [internet_service],
        'OnlineSecurity': ['No'],           # you can add more fields later
        'OnlineBackup': ['No'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['No'],
        'StreamingMovies': ['No'],
        'Contract': [contract],
        'PaperlessBilling': [paperless],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges]
    })

    # One-hot encode exactly like training
    input_dummies = pd.get_dummies(input_data, drop_first=True)

    # Make sure columns match training (add missing ones with 0)
    input_dummies = input_dummies.reindex(columns=model_columns, fill_value=0)

    # Scale the two numeric columns
    numeric_cols = ['tenure', 'MonthlyCharges']
    input_dummies[numeric_cols] = scaler.transform(input_dummies[numeric_cols])

    # Predict
    prediction = model.predict(input_dummies)[0]
    churn_prob = model.predict_proba(input_dummies)[0][1]

    # Show result
    if prediction == 1:
        st.error(f"**Prediction: WILL CHURN** (leave)")
        st.markdown(f"Churn probability: **{churn_prob:.1%}**")
    else:
        st.success(f"**Prediction: WILL STAY**")
        st.markdown(f"Churn probability: **{churn_prob:.1%}**")