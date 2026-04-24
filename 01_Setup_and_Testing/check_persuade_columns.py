import pandas as pd

df = pd.read_csv('data/raw/persuade_sampled.csv')
print("Columns in PERSUADE dataset:")
print(df.columns.tolist())
print("\nFirst row sample:")
print(df.iloc[0])