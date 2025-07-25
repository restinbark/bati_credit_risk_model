# 🏦 Credit Risk Prediction App  
**10 Academy Week 5 Challenge – End-to-End Machine Learning System**

This project builds a complete credit risk scoring pipeline using alternative customer data. It predicts whether a customer is **High Risk** or **Low Risk** based on transaction behavior and includes Streamlit app deployment for real-time predictions.

---

## 📌 Project Overview

The goal is to help a partner bank enable **Buy Now, Pay Later (BNPL)** functionality for digital customers by scoring risk using behavioral data (no credit history). The solution follows the entire ML lifecycle including:

- 📊 Business understanding & proxy label creation
- 🔍 Data preprocessing & feature engineering
- ⚙️ Model training & evaluation (Logistic Regression, Random Forest, XGBoost)
- 🧪 Model validation & testing
- 🚀 Streamlit app deployment

---

## 📘 Credit Scoring Business Understanding

### 🔐 Basel II Influence

Basel II emphasizes risk measurement and transparency. Hence, models must be **interpretable**, **auditable**, and **documented** — not just accurate. This guided the choice of feature engineering and model evaluation throughout the project.

### 🎯 Why Use a Proxy Label?

Since there's no direct "default" label in the data, we created a proxy using **Recency, Frequency, Monetary (RFM)** patterns and clustering. However, this comes with trade-offs:

- Potential to misclassify customers due to short-term inactivity
- Overfitting to behavioral patterns instead of broader risk
- Risk of rejecting good customers or accepting bad ones

### ⚖️ Trade-offs: Interpretable vs. Complex Models

| Criteria                | Logistic Regression + WoE     | Gradient Boosting (XGBoost)         |
|------------------------|-------------------------------|-------------------------------------|
| Interpretability       | ✅ High                        | ❌ Low (requires SHAP/LIME)         |
| Accuracy               | Moderate                      | ✅ High (captures non-linearities)  |
| Regulatory Preference  | ✅ Strong                      | Needs justification                 |
| Business Trust         | ✅ Easier to communicate       | ❌ Hard to explain predictions       |

---

## 🧱 Folder Structure

bati_credit_risk_model/
├── app.py # Streamlit app
├── models/
│ └── best_model.pkl # Trained XGBoost model
├── data/
│ ├── raw/ # Original input
│ └── processed/ # Cleaned & labeled
├── notebooks/
│ ├── 1.0-eda.ipynb
│ └── 2.0-modeling.ipynb
├── scripts/
│ ├── data_processing.py
│ ├── label_creator.py
│ └── generate_final_upload_data.py
├── requirements.txt
└── README.md


---

## 🚀 Streamlit App

---

## ⚙️ How to Run Locally

1. Clone the repository:  
   `git clone https://github.com/restinbark/bati_credit_risk_model.git`  
   `cd bati_credit_risk_model`

2. Create a virtual environment:  
   `python -m venv .venv`  
   - Windows: `.venv\Scripts\activate`  
   - Linux/macOS: `source .venv/bin/activate`

3. Install dependencies:  
   `pip install -r requirements.txt`

4. Launch the Streamlit app:  
   `streamlit run app.py`

---

## 📁 Input CSV Format

Upload a CSV with the following features:

Amount, Value, Hour, Month, Day, Weekday,
Amount_sum, Amount_mean, Amount_count, Amount_std


These are generated through feature engineering. Use the `generate_final_upload_data.py` script if needed.

---

## 📈 Model Performance (XGBoost)

- **Accuracy:** 99.6%  
- **Precision (High Risk):** 97%  
- **Recall (High Risk):** 98%  
- **F1-score (High Risk):** 97.5%

Best model saved as:  
`models/best_model.pkl`

---

## 🙋‍♂️ Author

**Barkilign Mulatu**    
🏁 Built as part of **10 Academy Week 5**

---

## ✅ License

MIT License — open to use, modify, and extend with credit.
