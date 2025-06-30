# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/best_model.pkl")

st.title("Credit Risk Prediction App")

st.write("Upload a CSV file or fill out the form below to predict if a customer is high risk.")

# Option 1: File upload
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    pred = model.predict(df)
    st.write("### Prediction Results")
    df["Prediction"] = pred
    df["Risk Level"] = df["Prediction"].map({0: "Low Risk", 1: "High Risk"})
    st.dataframe(df[["Risk Level"]])

# Option 2: Manual form input (for 1 customer)
with st.expander("Or manually input customer data"):
    amount = st.number_input("Amount")
    value = st.number_input("Value")
    hour = st.slider("Hour", 0, 23)
    month = st.slider("Month", 1, 12)
    day = st.slider("Day", 1, 31)
    weekday = st.slider("Weekday (0=Mon)", 0, 6)
    amount_sum = st.number_input("Amount Sum")
    amount_mean = st.number_input("Amount Mean")
    amount_count = st.number_input("Amount Count")
    amount_std = st.number_input("Amount Std")

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            "Amount": amount,
            "Value": value,
            "Hour": hour,
            "Month": month,
            "Day": day,
            "Weekday": weekday,
            "Amount_sum": amount_sum,
            "Amount_mean": amount_mean,
            "Amount_count": amount_count,
            "Amount_std": amount_std
        }])
        prediction = model.predict(input_df)[0]
        st.success("Prediction: **{}**".format("High Risk" if prediction == 1 else "Low Risk"))
