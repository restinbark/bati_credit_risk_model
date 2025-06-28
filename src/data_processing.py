
### Task 3 — Building the Pipeline


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


### Custom Transformer to Add Features

class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Convert time column
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

        # Time-based features
        df['Hour'] = df['TransactionStartTime'].dt.hour
        df['Month'] = df['TransactionStartTime'].dt.month
        df['Day'] = df['TransactionStartTime'].dt.day
        df['Weekday'] = df['TransactionStartTime'].dt.weekday

        # Customer-level aggregates
        customer_agg = df.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'count', 'std']
        })

        # Flatten MultiIndex
        customer_agg.columns = ['Amount_sum', 'Amount_mean', 'Amount_count', 'Amount_std']
        customer_agg.reset_index(inplace=True)

        # Merge aggregates back to original df
        df = df.merge(customer_agg, on='CustomerId', how='left')

        return df


### Categorical and Numeric Pipelines

def build_pipeline():
    categorical_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']
    numeric_cols = ['Amount', 'Value', 'Hour', 'Month', 'Day', 'Weekday', 'Amount_sum', 'Amount_mean', 'Amount_count', 'Amount_std']

    # Pipelines
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Column transformer
    preprocessor = ColumnTransformer(transformers=[
        ('cat', cat_pipeline, categorical_cols),
        ('num', num_pipeline, numeric_cols)
    ])

    # Full pipeline
    full_pipeline = Pipeline(steps=[
        ('feature_gen', FeatureGenerator()),
        ('preprocessor', preprocessor)
    ])

    return full_pipeline

###  Processing Function

def process_data(df):
    pipeline = build_pipeline()
    return pipeline.fit_transform(df)


