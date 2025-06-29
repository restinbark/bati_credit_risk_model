import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

        # Time features
        df['Hour'] = df['TransactionStartTime'].dt.hour
        df['Month'] = df['TransactionStartTime'].dt.month
        df['Day'] = df['TransactionStartTime'].dt.day
        df['Weekday'] = df['TransactionStartTime'].dt.weekday

        # Recency (for RFM / task 4)
        last_date = df['TransactionStartTime'].max()
        recency = df.groupby('CustomerId')['TransactionStartTime'].max().apply(lambda x: (last_date - x).days)
        recency.name = 'Recency'
        df = df.merge(recency, on='CustomerId', how='left')

        # Aggregates
        customer_agg = df.groupby('CustomerId')['Amount'].agg(['sum', 'mean', 'count', 'std']).reset_index()
        customer_agg.columns = ['CustomerId', 'Amount_sum', 'Amount_mean', 'Amount_count', 'Amount_std']
        df = df.merge(customer_agg, on='CustomerId', how='left')

        return df

def build_pipeline():
    categorical_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']
    numeric_cols = ['Amount', 'Value', 'Hour', 'Month', 'Day', 'Weekday',
                    'Recency', 'Amount_sum', 'Amount_mean', 'Amount_count', 'Amount_std']

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, categorical_cols),
        ('num', num_pipeline, numeric_cols)
    ])

    pipeline = Pipeline([
        ('feature_gen', FeatureGenerator()),
        ('preprocessor', preprocessor)
    ])

    return pipeline

def build_feature_matrix(df):
    pipeline = build_pipeline()
    return pipeline.fit_transform(df)
