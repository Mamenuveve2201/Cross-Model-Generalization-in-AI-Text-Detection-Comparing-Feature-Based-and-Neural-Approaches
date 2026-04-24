import pandas as pd

# Load the file with topics
persuade_df = pd.read_csv('data/raw/persuade_with_topics.csv')

print("✅ Verifying topics are loaded correctly:\n")

# Show first 10 essay topics
for idx in range(10):
    essay_text = persuade_df.iloc[idx]['full_text'][:80]
    topic = persuade_df.iloc[idx]['extracted_topic']
    
    print(f"Essay {idx}:")
    print(f"  Text: '{essay_text}...'")
    print(f"  Topic: '{topic}'")
    print()

print("\n🎯 The GPT and Claude scripts will use these exact topics!")
print("\nFor example, Essay 0 prompt will be:")
topic_0 = persuade_df.iloc[0]['extracted_topic']
length_0 = len(persuade_df.iloc[0]['full_text'])
print(f'  "Write an argumentative essay about {topic_0} that is approximately {length_0} characters long."')