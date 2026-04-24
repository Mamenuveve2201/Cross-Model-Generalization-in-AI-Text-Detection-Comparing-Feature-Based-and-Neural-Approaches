import pandas as pd

print("Loading Qwen samples...")
df = pd.read_csv('data/raw/qwen_samples.csv')

print(f"Total essays: {len(df)}")

# Count before
before = df['generated_text'].astype(str).str.contains('<think>', case=False).sum()
print(f"Essays with <think> BEFORE: {before}\n")

print("Cleaning all essays...\n")

# Clean each essay - very simple approach
for idx in range(len(df)):
    text = str(df.at[idx, 'generated_text'])
    
    # Find where </think> ends and take everything after it
    if '<think>' in text.lower():
        # Find the closing tag
        end_pos = text.lower().find('</think>')
        if end_pos != -1:
            # Take everything after </think>
            cleaned = text[end_pos + 8:].strip()  # 8 = len('</think>')
            df.at[idx, 'generated_text'] = cleaned
            df.at[idx, 'actual_length'] = len(cleaned)
            
            if idx < 5:
                print(f"Cleaned essay {idx}")
    
    if (idx + 1) % 200 == 0:
        print(f"Processed {idx + 1}/1000...")

# Save
df.to_csv('data/raw/qwen_samples.csv', index=False)

# Count after
after = df['generated_text'].astype(str).str.contains('<think>', case=False).sum()

print(f"\n✅ Done!")
print(f"Essays with <think> AFTER: {after}")
print(f"Successfully cleaned: {before - after} essays")