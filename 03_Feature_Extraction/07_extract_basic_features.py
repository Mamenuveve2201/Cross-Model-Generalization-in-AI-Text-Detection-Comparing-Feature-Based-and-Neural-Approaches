import pandas as pd
import spacy
from collections import Counter

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

print("Loading master dataset...")
df = pd.read_csv('data/raw/master_dataset.csv')

print(f"Total samples: {len(df)}\n")

# Test on first 5 samples
print("Testing feature extraction on 5 samples...\n")

def extract_features(text):
    """Extract voice, tense, and pronoun features from text"""
    
    doc = nlp(text)
    
    # Count passive and active voice
    passive_count = 0
    active_count = 0
    
    for token in doc:
        if token.dep_ == "nsubjpass":  # Passive voice indicator
            passive_count += 1
        elif token.dep_ == "nsubj":  # Active voice indicator
            active_count += 1
    
    # Count verb tenses
    past_tense = sum(1 for token in doc if token.tag_ == "VBD")  # Past tense
    present_tense = sum(1 for token in doc if token.tag_ in ["VBZ", "VBP"])  # Present
    future_tense = sum(1 for token in doc if token.text.lower() == "will")  # Future marker
    
    # Count pronouns by person
    first_person = sum(1 for token in doc if token.text.lower() in ["i", "me", "my", "mine", "we", "us", "our", "ours"])
    second_person = sum(1 for token in doc if token.text.lower() in ["you", "your", "yours"])
    third_person = sum(1 for token in doc if token.text.lower() in ["he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"])
    
    # Total words for normalization
    total_words = len([token for token in doc if not token.is_punct and not token.is_space])
    
    return {
        'passive_voice_freq': passive_count / total_words if total_words > 0 else 0,
        'active_voice_freq': active_count / total_words if total_words > 0 else 0,
        'past_tense_freq': past_tense / total_words if total_words > 0 else 0,
        'present_tense_freq': present_tense / total_words if total_words > 0 else 0,
        'future_tense_freq': future_tense / total_words if total_words > 0 else 0,
        'first_person_freq': first_person / total_words if total_words > 0 else 0,
        'second_person_freq': second_person / total_words if total_words > 0 else 0,
        'third_person_freq': third_person / total_words if total_words > 0 else 0
    }

# Test on first 5 samples
for idx in range(5):
    text = df.iloc[idx]['text']
    features = extract_features(text)
    
    print(f"Sample {idx} ({df.iloc[idx]['model_source']}):")
    print(f"  Passive voice: {features['passive_voice_freq']:.4f}")
    print(f"  Active voice: {features['active_voice_freq']:.4f}")
    print(f"  Past tense: {features['past_tense_freq']:.4f}")
    print(f"  Present tense: {features['present_tense_freq']:.4f}")
    print(f"  1st person: {features['first_person_freq']:.4f}")
    print(f"  2nd person: {features['second_person_freq']:.4f}")
    print(f"  3rd person: {features['third_person_freq']:.4f}")
    print()

print("✅ Feature extraction test complete!")
print("\nFeatures look good? If yes, we'll run on all 5,981 samples next.")