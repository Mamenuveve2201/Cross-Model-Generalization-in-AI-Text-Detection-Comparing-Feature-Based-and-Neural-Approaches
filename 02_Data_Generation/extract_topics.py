import pandas as pd
import re

# Load PERSUADE
persuade_df = pd.read_csv('data/raw/persuade_sampled.csv')

topics = []

print("Extracting topics from essays...\n")

for idx, row in persuade_df.iterrows():
    essay_text = row['full_text']
    
    # Get first 200 characters to analyze
    intro = essay_text[:200].lower()
    
    # Common PERSUADE topics (these are typical argumentative essay topics)
    # We'll match keywords to assign topics
    if 'positive' in intro or 'attitude' in intro or 'optimis' in intro:
        topic = "the importance of having a positive attitude"
    elif 'technology' in intro or 'computer' in intro or 'internet' in intro:
        topic = "the impact of technology on society"
    elif 'education' in intro or 'school' in intro or 'learn' in intro:
        topic = "the value of education"
    elif 'environment' in intro or 'climate' in intro or 'pollution' in intro:
        topic = "environmental conservation and climate change"
    elif 'social media' in intro or 'facebook' in intro or 'instagram' in intro:
        topic = "the effects of social media"
    elif 'success' in intro or 'achieve' in intro or 'goal' in intro:
        topic = "what it takes to achieve success"
    elif 'friend' in intro or 'relationship' in intro:
        topic = "the importance of friendship and relationships"
    elif 'exercise' in intro or 'health' in intro or 'fitness' in intro:
        topic = "the benefits of physical exercise and healthy living"
    else:
        # Generic fallback
        topic = "a topic of personal interest"
    
    topics.append(topic)
    
    if idx < 10:  # Show first 10
        print(f"Essay {idx}: '{essay_text[:80]}...' → Topic: {topic}")

# Add topics to dataframe
persuade_df['extracted_topic'] = topics
persuade_df.to_csv('data/raw/persuade_with_topics.csv', index=False)

print(f"\n✅ Extracted topics for {len(topics)} essays")
print("Saved to: data/raw/persuade_with_topics.csv")