import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.label_creator import attach_label_to_dataset

data = {
    'TransactionStartTime': ['2023-01-01', '2023-01-02', '2023-01-05'],
    'CustomerId': ['c1', 'c1', 'c2'],
    'TransactionId': [1, 2, 3],
    'Amount': [100, 150, 50]
}
df = pd.DataFrame(data)

labeled_df = attach_label_to_dataset(df)
print(labeled_df[['CustomerId', 'is_high_risk']])
