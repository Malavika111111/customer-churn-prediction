import streamlit as st
import pandas as pd
import joblib  

# Loading trained model & scaler
rf_model = joblib.load("random_forest.pkl")  
scaler = joblib.load("scaler0.pkl")

# Loading the dataset to get training feature names 
df = pd.read_excel("Churn (1) (2).xlsx")

# Drop unnecessary columns like 'Unnamed: 0'
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Applying the One-Hot Encoding same as training
df = pd.get_dummies(df, columns=['state', 'area.code'], drop_first=True)

# Store feature names used in training
training_columns = df.drop(columns=['churn']).columns

# Function to get the user input
def get_user_input():
    with st.sidebar:
        st.header("User Input Parameters")                               # Sidebar Title
        states = ['CA', 'NY', 'TX', 'FL', 'OH', 'MI', 'NJ', 'WA', 'VA']  # Example states
        area_codes = [408, 415, 510, 650, 708]                           # Example area codes

        state = st.selectbox('State', states)
        area_code = st.selectbox('Area Code', area_codes)

        account_length = st.slider('Account Length', 1, 500, 100)
        voice_plan = st.radio('Voice Plan', ['Yes', 'No'])
        voice_messages = st.slider('Voice Messages', 0, 500, 10)
        intl_plan = st.radio('International Plan', ['Yes', 'No'])
        intl_mins = st.slider('International Minutes', 0, 500, 20)
        intl_calls = st.slider('International Calls', 0, 100, 5)
        intl_charge = st.slider('International Charge', 0.0, 100.0, 2.5)

        day_mins = st.slider('Day Minutes', 0, 500, 180)
        day_calls = st.slider('Day Calls', 0, 100, 40)
        day_charge = st.slider('Day Charge', 0.0, 100.0, 20.5)

        eve_mins = st.slider('Evening Minutes', 0, 500, 200)
        eve_calls = st.slider('Evening Calls', 0, 100, 50)
        eve_charge = st.slider('Evening Charge', 0.0, 100.0, 18.7)

        night_mins = st.slider('Night Minutes', 0, 500, 250)
        night_calls = st.slider('Night Calls', 0, 100, 60)
        night_charge = st.slider('Night Charge', 0.0, 100.0, 15.2)

        customer_calls = st.slider('Customer Calls', 0, 500, 3)

    # Convert the user input into DataFrame
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

    # Applying One-Hot Encoding to ensure it matches training data
    user_input = pd.get_dummies(user_input, columns = ['state', 'area.code'], drop_first=True)

    # Fill missing numerical values with mean and categorical (one-hot) values with 0
    user_input = user_input.fillna(df.mean(numeric_only=True)).fillna(0)

    # Reindex to ensure all training columns exist
    user_input = user_input.reindex(columns = training_columns)

    # Convert DataFrame to NumPy array for scaling
    user_input_scaled = scaler.transform(user_input)
    return user_input_scaled

# Streamlit UI
st.title("Customer Churn Prediction")
st.write("Adjust the values in the **sidebar** to see predictions.")

# Calling the function get_user_input()
user_input_scaled = get_user_input()

# Predicting and displaying the result
prediction = int(rf_model.predict(user_input_scaled)[0])
if prediction == 1:
    st.write("The customer is **likely to churn**.")
else:
    st.write("The customer is **not likely to churn**.")

# Feature Importance Visualization
st.subheader("Feature Importance in Prediction")

# SHAP Explanation
explainer = shap.TreeExplainer(rf_model)  # Explainer for Random Forest
shap_values = explainer.shap_values(user_input_scaled)

# SHAP Summary Plot
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, user_input, feature_names=training_columns)
st.pyplot()

# Top Factors Bar Plot
st.subheader("Top Factors Influencing This Prediction")

# Get absolute SHAP values for sorting
shap_abs = np.abs(shap_values[0])
top_features = np.argsort(shap_abs)[-5:]  # Get top 5 influencing features

plt.figure(figsize=(8, 5))
plt.barh(np.array(training_columns)[top_features], shap_abs[top_features], color="royalblue")
plt.xlabel("Impact on Prediction")
plt.ylabel("Feature")
plt.title("Top 5 Features Affecting Churn Decision")
st.pyplot()
