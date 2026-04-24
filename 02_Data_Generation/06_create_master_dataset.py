import pandas as pd

print("="*60)
print("CREATING MASTER DATASET")
print("="*60)

# STEP 1: Load human essays
print("\n1. Loading human essays...")
human_df = pd.read_csv('data/raw/persuade_sampled.csv')
human_df = human_df[['text_id', 'full_text']].copy()
human_df.rename(columns={'text_id': 'essay_id', 'full_text': 'text'}, inplace=True)
human_df['label'] = 0  # 0 = human
human_df['model_source'] = 'human'
print(f"   ✅ Loaded {len(human_df)} human essays")

# STEP 2: Load AI essays from all 5 models
ai_models = {
    'gpt4o': 'data/raw/gpt4o_samples.csv',
    'claude': 'data/raw/claude_samples.csv',
    'gemini': 'data/raw/gemini_samples.csv',
    'llama': 'data/raw/llama_samples.csv',
    'qwen': 'data/raw/qwen_samples.csv'
}

all_dfs = [human_df]

print("\n2. Loading AI essays...")
for model_name, filepath in ai_models.items():
    df = pd.read_csv(filepath)
    
    # Keep only essay_id and generated_text
    df = df[['essay_id', 'generated_text']].copy()
    df.rename(columns={'generated_text': 'text'}, inplace=True)
    
    # Remove failures
    before_count = len(df)
    df = df[~df['text'].str.contains('GENERATION_FAILED', na=False, case=False)]
    df = df[~df['text'].str.contains('Please provide', na=False, case=False)]
    df = df[~df['text'].str.contains('I cannot', na=False, case=False)]
    after_count = len(df)
    
    # Add labels
    df['label'] = 1  # 1 = AI
    df['model_source'] = model_name
    
    all_dfs.append(df)
    
    removed = before_count - after_count
    print(f"   ✅ {model_name}: {after_count} essays (removed {removed} failures)")

# STEP 3: Combine all data
print("\n3. Combining all datasets...")
master_df = pd.concat(all_dfs, ignore_index=True)

# STEP 4: Verify and clean
print("\n4. Cleaning and verifying...")

# Remove extremely short essays (< 100 chars)
before_len = len(master_df)
master_df = master_df[master_df['text'].str.len() > 100]
after_len = len(master_df)
print(f"   Removed {before_len - after_len} very short essays")

# STEP 5: Save master dataset
master_df.to_csv('data/raw/master_dataset.csv', index=False)

print("\n" + "="*60)
print("✅ MASTER DATASET CREATED!")
print("="*60)
print(f"\nTotal samples: {len(master_df)}")
print(f"Shape: {master_df.shape}")
print(f"\nBreakdown by source:")
print(master_df['model_source'].value_counts().sort_index())
print(f"\nLabel distribution:")
print(f"  Human (0): {(master_df['label'] == 0).sum()}")
print(f"  AI (1): {(master_df['label'] == 1).sum()}")
print(f"\nSaved to: data/raw/master_dataset.csv")