import streamlit as st
import pandas as pd
import joblib  # Load trained model
import numpy as np

# Load trained model & scaler
rf_model = joblib.load("random_forest.pkl")  
scaler = joblib.load("scaler0.pkl")

# Load dataset to get training feature names (ensure index column is not included)
df = pd.read_excel("Churn (1) (2).xlsx")

# Drop unnecessary columns like 'Unnamed: 0' (if exists)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# One-Hot Encoding (same as training)
df = pd.get_dummies(df, columns=['state', 'area.code'], drop_first=True)

# Store feature names used in training
training_columns = df.drop(columns=['churn']).columns

# Function to get user input
def get_user_input():
    states = ['CA', 'NY', 'TX', 'FL', 'OH', 'MI', 'NJ', 'WA', 'VA']  # Example states
    area_codes = [408, 415, 510, 650, 708]  # Example area codes

    state = st.selectbox('State', states)
    area_code = st.selectbox('Area Code', area_codes)

    account_length = st.select_slider('Account Length', options=range(1, 501), value=100)
    voice_plan = st.selectbox('Voice Plan', ['Yes', 'No'])
    voice_messages = st.select_slider('Voice Messages', options=range(0, 501), value=10)
    intl_plan = st.selectbox('International Plan', ['Yes', 'No'])
    intl_mins = st.select_slider('International Minutes', options=range(0, 501), value=20)
    intl_calls = st.select_slider('International Calls', options=range(0, 101), value=5)
    intl_charge = st.select_slider('International Charge', options=[round(i * 0.1, 1) for i in range(0, 1001)], value=2.5)

    day_mins = st.select_slider('Day Minutes', options=range(0, 501), value=180)
    day_calls = st.select_slider('Day Calls', options=range(0, 101), value=40)
    day_charge = st.select_slider('Day Charge', options=[round(i * 0.1, 1) for i in range(0, 1001)], value=20.5)

    eve_mins = st.select_slider('Evening Minutes', options=range(0, 501), value=200)
    eve_calls = st.select_slider('Evening Calls', options=range(0, 101), value=50)
    eve_charge = st.select_slider('Evening Charge', options=[round(i * 0.1, 1) for i in range(0, 1001)], value=18.7)

    night_mins = st.select_slider('Night Minutes', options=range(0, 501), value=250)
    night_calls = st.select_slider('Night Calls', options=range(0, 101), value=60)
    night_charge = st.select_slider('Night Charge', options=[round(i * 0.1, 1) for i in range(0, 1001)], value=15.2)

    customer_calls = st.select_slider('Customer Calls', options=range(0, 501), value=3)

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
    missing_cols = set(training_columns) - set(user_input.columns)
    for col in missing_cols:
        user_input[col] = 0  # Add missing feature columns with 0

    # Ensure column order matches training data
    user_input = user_input.reindex(columns=training_columns, fill_value=0)

    # Convert DataFrame to NumPy array for scaling
    user_input_scaled = scaler.transform(user_input)

    return user_input_scaled

# Streamlit UI
st.title("Customer Churn Prediction")
user_input_scaled = get_user_input()

# Predict and display result
prediction = int(rf_model.predict(user_input_scaled)[0])
if prediction == 1:
    st.write("The customer is **likely to churn**. 🚨")
else:
    st.write("The customer is **not likely to churn**. ✅")
