import streamlit as st
import joblib
import numpy as np

# Load Model
model = joblib.load("loan_approval_model.pkl")

# App Title
st.title("🏦 Loan Approval Prediction App")

# Input Fields
gender = st.selectbox("Gender", [1, 0])
married = st.selectbox("Married", [1, 0])
dependents = st.number_input("Dependents", min_value=0, max_value=3)
education = st.selectbox("Education", [1, 0])
self_employed = st.selectbox("Self Employed", [1, 0])
applicant_income = st.number_input("Applicant Income")
coapplicant_income = st.number_input("Coapplicant Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Amount Term")
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", [2, 1, 0])

# Predict Function
if st.button("Predict Loan Approval"):
    features = np.array([[gender, married, dependents, education, self_employed, 
                          applicant_income, coapplicant_income, loan_amount, 
                          loan_term, credit_history, property_area]])
    prediction = model.predict(features)
    result = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
    st.success(f"Loan Status: {result}")
