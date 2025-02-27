import streamlit as st
import pandas as pd
import joblib  # Load trained model

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
    with st.sidebar:
    st.header("User Input Features")
    st.markdown("<div style='height: 600px; overflow-y: auto;'>", unsafe_allow_html=True)

    state = st.selectbox('State', ['CA', 'NY', 'TX', 'FL'])  # Example states
    area_code = st.number_input('Area Code', min_value=100, max_value=999, step=1, value=408)
    account_length = st.number_input('Account Length', min_value=1, max_value=500, value=100)

    # Plans
    st.subheader("Plans")
    voice_plan = st.selectbox('Voice Plan', ['Yes', 'No'])
    intl_plan = st.selectbox('International Plan', ['Yes', 'No'])

    # International Usage
    st.subheader("International Usage")
    intl_mins = st.number_input('International Minutes', min_value=0, max_value=500, value=20)
    intl_calls = st.number_input('International Calls', min_value=0, max_value=100, value=5)
    intl_charge = st.number_input('International Charge', min_value=0.0, max_value=100.0, value=2.5)

    # Day Usage
    st.subheader("Day Usage")
    day_mins = st.number_input('Day Minutes', min_value=0, max_value=500, value=180)
    day_calls = st.number_input('Day Calls', min_value=0, max_value=100, value=40)
    day_charge = st.number_input('Day Charge', min_value=0.0, max_value=100.0, value=20.5)

    # Evening Usage
    st.subheader("Evening Usage")
    eve_mins = st.number_input('Evening Minutes', min_value=0, max_value=500, value=200)
    eve_calls = st.number_input('Evening Calls', min_value=0, max_value=100, value=50)
    eve_charge = st.number_input('Evening Charge', min_value=0.0, max_value=100.0, value=18.7)

    # Night Usage
    st.subheader("Night Usage")
    night_mins = st.number_input('Night Minutes', min_value=0, max_value=500, value=250)
    night_calls = st.number_input('Night Calls', min_value=0, max_value=100, value=60)
    night_charge = st.number_input('Night Charge', min_value=0.0, max_value=100.0, value=15.2)

    # Customer Service Calls
    customer_calls = st.number_input('Customer Service Calls', min_value=0, max_value=500, value=3)

    st.markdown("</div>", unsafe_allow_html=True)


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
st.write("Adjust the values in the **sidebar** to see predictions.")

user_input_scaled = get_user_input()

# Predict and display result
prediction = int(rf_model.predict(user_input_scaled)[0])
if prediction == 1:
    st.write("### 🚨 The customer is **likely to churn**.")
else:
    st.write("### ✅ The customer is **not likely to churn**.")

