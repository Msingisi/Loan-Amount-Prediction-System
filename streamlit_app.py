import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Loan Amount Prediction", layout="centered")
st.title("🏦 Loan Sanction Amount Predictor")


# --- Expected columns from the model ---
expected_columns = [
    'Age', 'Income (USD)', 'Loan Amount Request (USD)', 'Current Loan Expenses (USD)',
    'Dependents', 'Credit Score', 'No. of Defaults', 'Property Age', 'Property Type',
    'Co-Applicant', 'Property Price',
    'Gender_F', 'Gender_M',
    'Income Stability_High', 'Income Stability_Low',
    'Profession_Commercial associate', 'Profession_Pensioner',
    'Profession_State servant', 'Profession_Working',
    'Location_Rural', 'Location_Semi-Urban', 'Location_Urban',
    'Expense Type 1_N', 'Expense Type 1_Y',
    'Expense Type 2_N', 'Expense Type 2_Y',
    'Has Active Credit Card_Active', 'Has Active Credit Card_Inactive', 'Has Active Credit Card_Unpossessed',
    'Property Location_Rural', 'Property Location_Semi-Urban', 'Property Location_Urban'
]

# --- Collect input ---
st.header("🔍 Applicant & Property Information")

age = st.slider("Age", 18, 75, 35)
income = st.number_input("Income (USD)", min_value=1000, max_value=200000, value=50000)
loan_request = st.number_input("Loan Amount Request (USD)", min_value=500, max_value=100000, value=15000)
loan_expenses = st.number_input("Current Loan Expenses (USD)", min_value=0, max_value=50000, value=2000)
dependents = st.slider("Dependents", 0, 5, 2)
credit_score = st.slider("Credit Score", 300, 900, 700)
property_type = st.slider("Property Type", 1, 4, 1)
property_age = st.slider("Property Age (Years)", 0.0, 50.0, 5.0)
property_price = st.number_input("Property Price", min_value=10000, max_value=1000000, value=120000)
defaults = st.radio("No. of Defaults", [0, 1])
co_applicant = st.radio("Co-Applicant Present?", [0, 1])

st.header("👤 Applicant Profile")

gender = st.radio("Gender", ["M", "F"])
income_stability = st.selectbox("Income Stability", ["High", "Low"])
profession = st.selectbox("Profession", ["Working", "State servant", "Pensioner", "Commercial associate"])
location = st.selectbox("Location", ["Urban", "Semi-Urban", "Rural"])
expense_1 = st.radio("Has Expense Type 1?", ["Y", "N"])
expense_2 = st.radio("Has Expense Type 2?", ["Y", "N"])
credit_card = st.selectbox("Active Credit Card", ["Active", "Inactive", "Unpossessed"])
property_location = st.selectbox("Property Location", ["Urban", "Semi-Urban", "Rural"])

# --- Build input dictionary ---
input_data = {
    'Age': age,
    'Income (USD)': income,
    'Loan Amount Request (USD)': loan_request,
    'Current Loan Expenses (USD)': loan_expenses,
    'Dependents': dependents,
    'Credit Score': credit_score,
    'No. of Defaults': defaults,
    'Property Age': property_age,
    'Property Type': property_type,
    'Co-Applicant': co_applicant,
    'Property Price': property_price,
    f'Gender_{gender}': 1,
    f'Income Stability_{income_stability}': 1,
    f'Profession_{profession}': 1,
    f'Location_{location}': 1,
    f'Expense Type 1_{expense_1}': 1,
    f'Expense Type 2_{expense_2}': 1,
    f'Has Active Credit Card_{credit_card}': 1,
    f'Property Location_{property_location}': 1,
}

# --- Fill missing columns ---
for col in expected_columns:
    if col not in input_data:
        input_data[col] = 0

# --- Format input for model ---
df = pd.DataFrame([input_data])[expected_columns]

# --- Predict ---
if st.button("Predict Loan Amount"):
    try:
        response = requests.post(
            url="http://127.0.0.1:5005/invocations",
            headers={"Content-Type": "application/json"},
            json={"dataframe_records": df.to_dict(orient="records")}
        )
        if response.status_code == 200:
            prediction = response.json()["predictions"][0]
            st.success(f"Predicted Loan Sanction Amount: **${prediction:,.2f}**")
        else:
            st.error(f"Error: {response.status_code}\n{response.text}")
    except Exception as e:
        st.error(f"Failed to connect to MLflow model server.\n{str(e)}")

# --- Footer ---
st.markdown("---")
st.markdown("🔧 **MLflow Model Server:** `http://127.0.0.1:5005/invocations`")