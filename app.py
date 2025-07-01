# app.py

import streamlit as st
import pandas as pd
import joblib
import os

# Page setup
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("💳 Credit Risk Prediction App")
st.markdown("Upload a CSV file to predict if customers are **High Risk** or **Low Risk**.")

# Load model
@st.cache_resource
def load_model():
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found at 'models/best_model.pkl'")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# Set expected number of features (based on transformed final_upload_data.csv)
expected_num_cols = 46  # <-- change this if your model was trained on a different number

# File upload
uploaded_file = st.file_uploader("📁 Upload your input CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Check the shape
        if df.shape[1] != expected_num_cols:
            st.error(f"❌ Uploaded file must have exactly {expected_num_cols} columns. Your file has {df.shape[1]}.")
            st.stop()

        # Predict
        preds = model.predict(df)
        df["Risk_Prediction"] = preds
        df["Risk_Level"] = df["Risk_Prediction"].map({0: "Low Risk", 1: "High Risk"})

        st.success("✅ Prediction complete!")
        st.write(df[["Risk_Level"]].head())

        # Option to download results
        csv_out = df.to_csv(index=False)
        st.download_button("⬇️ Download Predictions as CSV", csv_out, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ An error occurred while processing the file: {e}")

else:
    st.info("Please upload a file to get started.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit | 10 Academy Week 5")
