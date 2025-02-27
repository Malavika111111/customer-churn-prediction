import streamlit as st
import pandas as pd
import joblib  # Load trained model
import numpy as np

# Load trained model & scaler
rf_model = joblib.load("random_forest.pkl")  
scaler = joblib.load("scaler0.pkl")

# Load dataset to get training feature names
df = pd.read_excel("Churn (1) (2).xlsx")

# One-Hot Encoding (same as training)
df = pd.get_dummies(df, columns=['state', 'area.code'], drop_first=True)

# Store feature names used in training
training_columns = df.drop(columns=['churn']).columns.tolist()

# Function to get user input
def get_user_input():
    state = st.selectbox('State', ['CA', 'NY', 'TX', 'FL'])  # Example states
    area_code = st.number_input('Area Code', min_value=100, max_value=999, step=1, value=408)
    account_length = st.number_input('Account Length', min_value=1, max_value=500, value=100)
    voice_plan = st.selectbox('Voice Plan', ['Yes', 'No'])
    voice_messages = st.number_input('Voice Messages', min_value=0, max_value=500, value=10)
    intl_plan = st.selectbox('International Plan', ['Yes', 'No'])
    intl_mins = st.number_input('International Minutes', min_value=0, max_value=500, value=20)
    intl_calls = st.number_input('International Calls', min_value=0, max_value=100, value=5)
    intl_charge = st.number_input('International Charge', min_value=0.0, max_value=100.0, value=2.5)
    day_mins = st.number_input('Day Minutes', min_value=0, max_value=500, value=180)
    day_calls = st.number_input('Day Calls', min_value=0, max_value=100, value=40)
    day_charge = st.number_input('Day Charge', min_value=0.0, max_value=100.0, value=20.5)
    eve_mins = st.number_input('Evening Minutes', min_value=0, max_value=500, value=200)
    eve_calls = st.number_input('Evening Calls', min_value=0, max_value=100, value=50)
    eve_charge = st.number_input('Evening Charge', min_value=0.0, max_value=100.0, value=18.7)
    night_mins = st.number_input('Night Minutes', min_value=0, max_value=500, value=250)
    night_calls = st.number_input('Night Calls', min_value=0, max_value=100, value=60)
    night_charge = st.number_input('Night Charge', min_value=0.0, max_value=100.0, value=15.2)
    customer_calls = st.number_input('Customer Calls', min_value=0, max_value=500, value=3)

    # Convert user input into DataFrame
    user_input = pd.DataFrame({
        'state': [state],
        'area.code': [area_code],
        'account.length': [account_length],
        'voice.plan': [1 if voice_plan == 'Yes' else 0],
        'voice.messages': [voice_messages],
        'intl.plan': [1 if intl_plan == 'Yes' else 0],
        'intl.mins': [intl_mins],
        'intl.calls': [intl_calls],
        'intl.charge': [intl_charge],
        'day.mins': [day_mins],
        'day.calls': [day_calls],
        'day.charge': [day_charge],
        'eve.mins': [eve_mins],
        'eve.calls': [eve_calls],
        'eve.charge': [eve_charge],
        'night.mins': [night_mins],
        'night.calls': [night_calls],
        'night.charge': [night_charge],
        'customer.calls': [customer_calls]
    })

    # Apply One-Hot Encoding (Ensure it matches training)
    user_input = pd.get_dummies(user_input, columns=['state', 'area.code'], drop_first=True)

    # Add missing columns (if any)
    for col in training_columns:
        if col not in user_input.columns:
            user_input[col] = 0  # Fill missing feature columns with 0

    # Ensure column order matches training data
    user_input = user_input[training_columns]

    # Convert DataFrame to NumPy array for scaling
    user_input_scaled = scaler.transform(user_input)
    return user_input_scaled

# Streamlit UI
st.title("Customer Churn Prediction")

try:
    user_input_scaled = get_user_input()

    # Predict and display result
    prediction = int(rf_model.predict(user_input_scaled)[0])
    if prediction == 1:
        st.write("The customer is **likely to churn**. 🚨")
    else:
        st.write("The customer is **not likely to churn**. ✅")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
