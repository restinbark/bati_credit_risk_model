from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class FeatureGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['Hour'] = df['TransactionStartTime'].dt.hour
        df['Month'] = df['TransactionStartTime'].dt.month
        df['Day'] = df['TransactionStartTime'].dt.day
        df['Weekday'] = df['TransactionStartTime'].dt.weekday

        agg = df.groupby('CustomerId')['Amount'].agg(['sum', 'mean', 'count', 'std']).reset_index()
        agg.columns = ['CustomerId', 'Amount_sum', 'Amount_mean', 'Amount_count', 'Amount_std']
        df = df.merge(agg, on='CustomerId', how='left')
        return df


def build_pipeline():
    categorical_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']
    numeric_cols = ['Amount', 'Value', 'Hour', 'Month', 'Day', 'Weekday',
                    'Amount_sum', 'Amount_mean', 'Amount_count', 'Amount_std']

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('cat', cat_pipeline, categorical_cols),
        ('num', num_pipeline, numeric_cols)
    ])

    full_pipeline = Pipeline(steps=[
        ('feature_gen', FeatureGenerator()),
        ('preprocessor', preprocessor)
    ])

    return full_pipeline
