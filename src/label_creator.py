# src/label_creator.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def generate_rfm_label(df):
    """
    Generates proxy label 'is_high_risk' using RFM features + KMeans clustering
    """
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    last_date = df['TransactionStartTime'].max()

    # RFM feature extraction
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (last_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    }).reset_index()

    # Normalize RFM
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=42)

    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

    # Label the riskiest cluster
    risky_cluster = rfm.groupby('cluster')['Monetary'].mean().idxmin()
    rfm['is_high_risk'] = (rfm['cluster'] == risky_cluster).astype(int)

    return rfm[['CustomerId', 'is_high_risk']]

def attach_label_to_dataset(df):
    """
    Adds is_high_risk column to original dataset
    """
    label_df = generate_rfm_label(df)
    return df.merge(label_df, on='CustomerId', how='left')
