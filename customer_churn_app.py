import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
rf_model = joblib.load("random_forest.pkl")
scaler = joblib.load("scaler0.pkl")

# Load training data to extract feature names
df = pd.read_excel("Churn (1) (2).xlsx")

# One-Hot Encoding for categorical columns (same as training)
df = pd.get_dummies(df, columns=['state', 'area.code'], drop_first=True)

# Get the feature names used during training
training_columns = df.drop(columns=['churn']).columns

# Function to get user input and preprocess it
def get_user_input():
    st.sidebar.header("Enter Customer Details")

    state = st.sidebar.selectbox('State', ['CA', 'NY', 'TX', 'FL'])  # Example states
    area_code = st.sidebar.number_input('Area Code', min_value=100, max_value=999, step=1, value=408)
    account_length = st.sidebar.number_input('Account Length', min_value=1, max_value=500, value=100)
    voice_plan = st.sidebar.selectbox('Voice Plan', ['Yes', 'No'])
    voice_messages = st.sidebar.number_input('Voice Messages', min_value=0, max_value=500, value=10)
    intl_plan = st.sidebar.selectbox('International Plan', ['Yes', 'No'])
    intl_mins = st.sidebar.number_input('International Minutes', min_value=0, max_value=500, value=20)
    intl_calls = st.sidebar.number_input('International Calls', min_value=0, max_value=100, value=5)
    intl_charge = st.sidebar.number_input('International Charge', min_value=0.0, max_value=100.0, value=2.5)
    day_mins = st.sidebar.number_input('Day Minutes', min_value=0, max_value=500, value=180)
    day_calls = st.sidebar.number_input('Day Calls', min_value=0, max_value=100, value=40)
    day_charge = st.sidebar.number_input('Day Charge', min_value=0.0, max_value=100.0, value=20.5)
    eve_mins = st.sidebar.number_input('Evening Minutes', min_value=0, max_value=500, value=200)
    eve_calls = st.sidebar.number_input('Evening Calls', min_value=0, max_value=100, value=50)
    eve_charge = st.sidebar.number_input('Evening Charge', min_value=0.0, max_value=100.0, value=18.7)
    night_mins = st.sidebar.number_input('Night Minutes', min_value=0, max_value=500, value=250)
    night_calls = st.sidebar.number_input('Night Calls', min_value=0, max_value=100, value=60)
    night_charge = st.sidebar.number_input('Night Charge', min_value=0.0, max_value=100.0, value=15.2)
    customer_calls = st.sidebar.number_input('Customer Service Calls', min_value=0, max_value=500, value=3)

    # Convert categorical inputs to numerical
    voice_plan = 1 if voice_plan == "Yes" else 0
    intl_plan = 1 if intl_plan == "Yes" else 0

    # Create dataframe
    user_input = pd.DataFrame({
        'state': [state],
        'area.code': [area_code],
        'account.length': [account_length],
        'voice.plan': [voice_plan],
        'voice.messages': [voice_messages],
        'intl.plan': [intl_plan],
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

    # One-Hot Encoding for categorical variables
    user_input = pd.get_dummies(user_input, columns=['state', 'area.code'], drop_first=True)

    # Ensure all missing columns are added
    missing_cols = set(training_columns) - set(user_input.columns)
    for col in missing_cols:
        user_input[col] = 0  # Add missing columns with default value 0

    # Reorder columns to match training data
    user_input = user_input[training_columns]

    # Scale input data
    user_input_scaled = scaler.transform(user_input)
    return user_input_scaled

# Streamlit UI
st.title("📊 Customer Churn Prediction App")
st.write("This app predicts whether a customer is likely to churn based on input features.")

# Get user input
user_input_scaled = get_user_input()

# Predict churn probability
prediction = rf_model.predict(user_input_scaled)[0]
prediction_prob = rf_model.predict_proba(user_input_scaled)[:, 1]

# Display results
st.subheader("Prediction Result")
if prediction == 1:
    st.error("The customer is **likely to churn**.")
else:
    st.success("The customer is **not likely to churn**.")

st.subheader("Churn Probability")
st.write(f" **{prediction_prob[0] * 100:.2f}%** chance of churn")

# Additional info
st.info("This model is based on a trained Random Forest classifier.")
