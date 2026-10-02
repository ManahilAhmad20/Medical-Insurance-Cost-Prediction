import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("insurance_model.pkl")

# Columns used when training the model
feature_columns = [
    'age',
    'bmi',
    'children',
    'sex_male',
    'smoker_yes',
    'region_northwest',
    'region_southeast',
    'region_southwest'
]

st.title("🏥 Medical Insurance Cost Predictor")

st.write("Enter your details below to estimate your medical insurance cost.")

# User inputs
age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

gender = st.selectbox(
    "Gender",
    ["female", "male"]
)

smoker = st.selectbox(
    "Smoking Status",
    ["no", "yes"]
)

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# Prediction
if st.button("Predict Insurance Cost"):

    input_data = pd.DataFrame([{
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex': gender,
        'smoker': smoker,
        'region': region
    }])

    # Convert categorical values into numbers
    input_data = pd.get_dummies(
        input_data,
        columns=['sex', 'smoker', 'region'],
        drop_first=True
    )

    # Make sure columns are exactly the same as training data
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated Insurance Cost: ${prediction:,.2f}"
    )
