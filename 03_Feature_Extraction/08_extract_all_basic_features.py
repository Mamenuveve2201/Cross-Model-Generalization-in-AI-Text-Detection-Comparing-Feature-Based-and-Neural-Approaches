import pandas as pd
import spacy
from tqdm import tqdm

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

print("Loading master dataset...")
df = pd.read_csv('data/raw/master_dataset.csv')

print(f"Total samples: {len(df)}\n")
print("Extracting basic features from all samples...")
print("This will take approximately 15-25 minutes.\n")

def extract_features(text):
    """Extract voice, tense, and pronoun features from text"""
    
    try:
        doc = nlp(text[:100000])  # Limit to avoid memory issues
        
        # Count passive and active voice
        passive_count = sum(1 for token in doc if token.dep_ == "nsubjpass")
        active_count = sum(1 for token in doc if token.dep_ == "nsubj")
        
        # Count verb tenses
        past_tense = sum(1 for token in doc if token.tag_ == "VBD")
        present_tense = sum(1 for token in doc if token.tag_ in ["VBZ", "VBP"])
        future_tense = sum(1 for token in doc if token.text.lower() == "will")
        
        # Count pronouns
        first_person = sum(1 for token in doc if token.text.lower() in ["i", "me", "my", "mine", "we", "us", "our", "ours"])
        second_person = sum(1 for token in doc if token.text.lower() in ["you", "your", "yours"])
        third_person = sum(1 for token in doc if token.text.lower() in ["he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"])
        
        total_words = len([token for token in doc if not token.is_punct and not token.is_space])
        
        if total_words == 0:
            return {f: 0.0 for f in ['passive_voice_freq', 'active_voice_freq', 'past_tense_freq', 
                                      'present_tense_freq', 'future_tense_freq', 'first_person_freq', 
                                      'second_person_freq', 'third_person_freq']}
        
        return {
            'passive_voice_freq': passive_count / total_words,
            'active_voice_freq': active_count / total_words,
            'past_tense_freq': past_tense / total_words,
            'present_tense_freq': present_tense / total_words,
            'future_tense_freq': future_tense / total_words,
            'first_person_freq': first_person / total_words,
            'second_person_freq': second_person / total_words,
            'third_person_freq': third_person / total_words
        }
    except Exception as e:
        print(f"Error processing text: {e}")
        return {f: 0.0 for f in ['passive_voice_freq', 'active_voice_freq', 'past_tense_freq', 
                                  'present_tense_freq', 'future_tense_freq', 'first_person_freq', 
                                  'second_person_freq', 'third_person_freq']}

# Extract features for all samples
features_list = []

for idx in tqdm(range(len(df)), desc="Extracting features"):
    text = df.iloc[idx]['text']
    features = extract_features(text)
    features_list.append(features)

# Convert to DataFrame
features_df = pd.DataFrame(features_list)

# Combine with original data
result_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

# Save
result_df.to_csv('data/raw/features_basic.csv', index=False)

print(f"\n✅ Feature extraction complete!")
print(f"Saved to: data/raw/features_basic.csv")
print(f"Shape: {result_df.shape}")
print(f"\nColumns added: {list(features_df.columns)}")