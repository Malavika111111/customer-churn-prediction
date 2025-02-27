import streamlit as st
import pandas as pd
import joblib  # Load trained model
import numpy as np

# Load trained model & scaler
rf_model = joblib.load("random_forest.pkl")  
scaler = joblib.load("scaler0.pkl")

# Load dataset to get training feature names
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
    st.sidebar.header("Customer Information")

    # State & Area Code Selection
    state = st.sidebar.selectbox("Select State", df['state'].unique())  # Get unique states from dataset
    area_code = st.sidebar.selectbox("Select Area Code", df['area.code'].unique())  # Get unique area codes

    account_length = st.sidebar.slider("Account Length (days)", 0, 365, 100)
    voice_plan = st.sidebar.selectbox("Has Voice Plan?", ["No", "Yes"])
    intl_plan = st.sidebar.selectbox("Has International Plan?", ["No", "Yes"])
    intl_mins = st.sidebar.slider("International Minutes", 0.0, 60.0, 15.0)
    intl_calls = st.sidebar.slider("International Calls", 0, 20, 5)
    intl_charge = intl_mins * 0.5  # Example: Charge calculation
    day_mins = st.sidebar.slider("Day Minutes", 0.0, 400.0, 150.0)
    day_calls = st.sidebar.slider("Day Calls", 0, 200, 100)
    day_charge = day_mins * 0.25  # Example charge formula
    customer_calls = st.sidebar.slider("Customer Service Calls", 0, 10, 2)

    # Convert user input into DataFrame
    user_input = pd.DataFrame({
        'state': [state],
        'area.code': [area_code],
        'account.length': [account_length],
        'voice.plan': [1 if voice_plan == 'Yes' else 0],
        'intl.plan': [1 if intl_plan == 'Yes' else 0],
        'intl.mins': [intl_mins],
        'intl.calls': [intl_calls],
        'intl.charge': [intl_charge],
        'day.mins': [day_mins],
        'day.calls': [day_calls],
        'day.charge': [day_charge],
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
