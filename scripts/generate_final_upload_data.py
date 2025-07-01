# scripts/generate_final_upload_data.py

import sys
import os
sys.path.append(os.path.abspath("src"))  # 👈 This lets Python find src/

import pandas as pd
import joblib
from data_processing import build_pipeline  # ✅ after adding src to path

# Load the raw labeled file
df = pd.read_csv("data/processed/credit_data_with_proxy_label.csv")

# Drop label columns before transforming (if present)
drop_cols = ['is_high_risk', 'FraudResult']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Load feature pipeline and transform
pipeline = build_pipeline()
X = pipeline.fit_transform(df)

# Save output (skip column names if OneHotEncoder used)
X_df = pd.DataFrame(X)
X_df.to_csv("data/processed/final_upload_data.csv", index=False)

print("✅ final_upload_data.csv created successfully.")
