# bati_credit_risk_model
# 🏦 Credit Risk Scoring Model

A machine learning pipeline to predict customer credit risk using alternative behavioral data. Built using Python, scikit-learn, MLflow, and FastAPI.

## 📚 Table of Contents
- [Credit Scoring Business Understanding](#-credit-scoring-business-understanding)
- [Data Exploration](#)
- [Feature Engineering](#)
- [Model Training](#)
- [API & Deployment](#)

# 🏦 Credit Risk Probability Model (10 Academy - Week 5)

## 📌 Project Overview

This project is part of the 10 Academy Week 5 challenge. The goal is to build a **credit risk scoring system** using alternative data from an e-commerce platform. The model predicts the probability that a customer is high-risk using transaction behavior, with the ultimate goal of enabling a **Buy Now, Pay Later (BNPL)** service for a partner bank.

The project follows the full machine learning lifecycle including:
- Business understanding
- Data preprocessing & feature engineering
- Modeling & hyperparameter tuning
- Model evaluation & tracking (MLflow)
- Model serving using FastAPI
- CI/CD using GitHub Actions and Docker

---

## 🧱 Folder Structure

bati-credit-risk-model/
├── .github/workflows/ci.yml # GitHub Actions workflow for CI/CD
├── data/
│ ├── raw/ # Raw data (ignored in .gitignore)
│ └── processed/ # Processed features for training
├── notebooks/
│ └── 1.0-eda.ipynb # Exploratory Data Analysis notebook
├── src/
│ ├── init.py
│ ├── data_processing.py # Feature engineering and preprocessing
│ ├── train.py # Model training script
│ ├── predict.py # Model inference script
│ └── api/
│ ├── main.py # FastAPI application
│ └── pydantic_models.py # Data validation for the API
├── tests/
│ └── test_data_processing.py # Unit tests
├── Dockerfile # Docker build file
├── docker-compose.yml # Docker orchestration
├── requirements.txt # Project dependencies
├── .gitignore # Files/folders to exclude from Git
└── README.md # Project documentation (this file)


---

## 📘 Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord highlights the importance of accurately measuring and managing credit risk. Since financial institutions must justify their credit decisions to regulators, models used in credit scoring must be interpretable, auditable, and explainable. A black-box model may perform well but is not acceptable if its decisions cannot be understood and verified. This pushes us to build models that are not only accurate but also transparent, well-documented, and compliant.

---

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In our dataset, we don’t have a label that directly says whether a customer defaulted. To build a model, we need a target variable—so we create a **proxy** using behavioral patterns like **Recency**, **Frequency**, and **Monetary (RFM)** values. However, this introduces risks:
- Customers might be misclassified as high-risk based on temporary inactivity.
- Over-reliance on transaction behavior may ignore other key risk factors.
- Bad proxies can lead to wrong credit decisions—either rejecting good customers or accepting risky ones.

Hence, we must carefully design and validate this proxy target.

---

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

| Criteria                    | Logistic Regression + WoE         | Gradient Boosting (e.g., XGBoost) |
|----------------------------|------------------------------------|-----------------------------------|
| **Interpretability**       | High – easily explainable          | Low – needs SHAP or LIME to interpret |
| **Performance**            | Moderate – may miss complex patterns | High – captures non-linear relationships |
| **Compliance**             | Strong – preferred by regulators   | Risky – requires extra validation |
| **Speed of Deployment**    | Fast                               | May be slower                     |
| **Business Trust**         | High                               | Lower unless explained well       |

In financial services, **simplicity and trust** often outweigh slight gains in accuracy.

---

✅ This understanding helps guide our model choice, feature design, and communication strategy in a real-world banking scenario.
