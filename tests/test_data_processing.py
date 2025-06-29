import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.data_processing import build_feature_matrix

# Sample test
data = {
    'TransactionStartTime': ['2023-01-01 12:00:00', '2023-01-02 13:30:00'],
    'CustomerId': ['cust1', 'cust2'],
    'Amount': [100, 150],
    'Value': [90, 130],
    'ProductCategory': ['airtime', 'data_bundle'],
    'ChannelId': ['app', 'ussd'],
    'PricingStrategy': [1, 2]
}
df = pd.DataFrame(data)

transformed = build_feature_matrix(df)
print(transformed.shape)
