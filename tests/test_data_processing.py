import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import process_data
import pandas as pd

df = pd.read_csv("data/raw/data.csv")
processed = process_data(df)
print(processed.shape)
