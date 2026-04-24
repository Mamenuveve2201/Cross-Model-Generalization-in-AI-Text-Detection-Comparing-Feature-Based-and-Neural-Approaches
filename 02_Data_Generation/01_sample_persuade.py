import pandas as pd
import random

print("Loading PERSUADE dataset...")
df = pd.read_csv('data/raw/train.csv')

print(f"Total essays available: {len(df)}")

# Randomly sample 1,000 essays
random.seed(42)  # For reproducibility
sampled_df = df.sample(n=1000, random_state=42)

# Save the sampled essays
sampled_df.to_csv('data/raw/persuade_sampled.csv', index=False)

print(f"✅ Sampled 1,000 essays saved to: data/raw/persuade_sampled.csv")
print(f"\nFirst essay preview:")
print(f"ID: {sampled_df.iloc[0]['text_id']}")
print(f"Text length: {len(sampled_df.iloc[0]['full_text'])} characters")
print(f"First 200 chars: {sampled_df.iloc[0]['full_text'][:200]}...")s