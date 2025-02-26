import streamlit as st
import pandas as pd
import joblib  # To load the trained model

# Load the trained model and scaler
rf_model = joblib.load("random_forest_model2.pkl")  # Ensure the correct model name
scaler = joblib.load("scaler2.pkl")

# Load the data used during model training
df = pd.read_excel("Churn (1) (2).xlsx")

# Prepare the features for training (same as during training)
binary_cols = ['voice.plan', 'intl.plan']
df[binary_cols] = df[binary_cols].apply(lambda col: col.map({"Yes": 1, "No": 0}))

# One-Hot Encoding for categorical columns during training (if done earlier)
df = pd.get_dummies(df, columns=['state', 'area.code'], drop_first=True)

# Get the feature names used during training
training_columns = df.drop(columns=['churn']).columns

# Function to get user input and preprocess it
def get_user_input():
    # Collecting user input through Streamlit widgets
    state = st.selectbox('State', ['CA', 'NY', 'TX', 'FL'])  # Example states
    area_code = st.number_input('Area Code', min_value=100, max_value=999, step=1)
    account_length = st.number_input('Account Length', min_value=1, max_value=500)
    voice_plan = st.selectbox('Voice Plan', ['Yes', 'No'])
    voice_messages = st.number_input('Voice Messages', min_value=0, max_value=500)
    intl_plan = st.selectbox('International Plan', ['Yes', 'No'])
    intl_mins = st.number_input('International Minutes', min_value=0, max_value=500)
    intl_calls = st.number_input('International Calls', min_value=0, max_value=100)
    intl_charge = st.number_input('International Charge', min_value=0, max_value=100)
    day_mins = st.number_input('Day Minutes', min_value=0, max_value=500)
    day_calls = st.number_input('Day Calls', min_value=0, max_value=100)
    day_charge = st.number_input('Day Charge', min_value=0.0, max_value=100.0)
    eve_mins = st.number_input('Evening Minutes', min_value=0, max_value=500)
    eve_calls = st.number_input('Evening Calls', min_value=0, max_value=100)
    eve_charge = st.number_input('Evening Charge', min_value=0.0, max_value=100.0)
    night_mins = st.number_input('Night Minutes', min_value=0, max_value=500)
    night_calls = st.number_input('Night Calls', min_value=0, max_value=100)
    night_charge = st.number_input('Night Charge', min_value=0.0, max_value=100.0)
    customer_calls = st.number_input('Customer Calls', min_value=0, max_value=500)

    # Create a DataFrame with user inputs
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

    # Debugging: Print out the columns of user_input and training_columns
    st.write("User input columns:", user_input.columns)
    st.write("Training columns:", training_columns)

    # Check if all training columns exist in user_input
    missing_columns = [col for col in training_columns if col not in user_input.columns]
    if missing_columns:
        st.write(f"Missing columns: {missing_columns}")

    # Re-order columns to match the training columns
    try:
        user_input = user_input[training_columns]
    except KeyError as e:
        st.write(f"Error: {e}")
        return None

    # Handle missing columns by adding them with 0 values if needed
    for col in training_columns:
        if col not in user_input.columns:
            user_input[col] = 0

    # Apply the same scaler used for training
    user_input = scaler.transform(user_input)  # Transform the input with the scaler
    return user_input

# Streamlit app UI
st.title("Customer Churn Prediction")

# Get the user input and predict the churn
user_input = get_user_input()
if user_input is not None:
    # Predict the churn probability
    prediction = rf_model.predict(user_input)
    
    # Display the prediction result
    if prediction == 1:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is not likely to churn.")
