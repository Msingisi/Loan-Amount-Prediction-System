import pandas as pd
import json
import requests

# 1. Define the expected columns your model was trained on
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

# 2. Create a sample test input row (fill only what's relevant)
sample_input = {
    'Age': 35.0,
    'Income (USD)': 50000.0,
    'Loan Amount Request (USD)': 15000.0,
    'Current Loan Expenses (USD)': 2000.0,
    'Dependents': 2.0,
    'Credit Score': 700.0,
    'No. of Defaults': 0.0,
    'Property Age': 5.0,
    'Property Type': 1.0,
    'Co-Applicant': 0.0,
    'Property Price': 120000.0,
    'Gender_M': 1, 'Gender_F': 0,
    'Income Stability_High': 1, 'Income Stability_Low': 0,
    'Profession_Working': 1, 'Profession_State servant': 0,
    'Profession_Pensioner': 0, 'Profession_Commercial associate': 0,
    'Location_Urban': 1, 'Location_Semi-Urban': 0, 'Location_Rural': 0,
    'Expense Type 1_Y': 1, 'Expense Type 1_N': 0,
    'Expense Type 2_Y': 1, 'Expense Type 2_N': 0,
    'Has Active Credit Card_Active': 1,
    'Has Active Credit Card_Inactive': 0,
    'Has Active Credit Card_Unpossessed': 0,
    'Property Location_Urban': 1, 'Property Location_Semi-Urban': 0, 'Property Location_Rural': 0
}

# 3. Ensure all expected columns are present, fill missing with 0
for col in expected_columns:
    if col not in sample_input:
        sample_input[col] = 0

# 4. Build the DataFrame aligned to the expected columns
df = pd.DataFrame([sample_input])[expected_columns]

# 5. Send request to MLflow model server
try:
    response = requests.post(
        url="http://127.0.0.1:5005/invocations",  # Port must match your MLflow server
        headers={"Content-Type": "application/json"},
        json={"dataframe_records": df.to_dict(orient="records")}
    )

    # 6. Show prediction
    if response.status_code == 200:
        print(" Prediction:", response.json())
    else:
        print(" Error:", response.status_code)
        print(response.text)

except Exception as e:
    print(" Failed to contact model server.")
    print(str(e))