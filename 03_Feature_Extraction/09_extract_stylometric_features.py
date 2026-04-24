import pandas as pd
import spacy
import numpy as np
from collections import Counter
from tqdm import tqdm
import math

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

print("Loading basic features dataset...")
df = pd.read_csv('data/raw/features_basic.csv')

print(f"Total samples: {len(df)}\n")
print("Extracting stylometric features from all samples...")
print("This will take approximately 20-30 minutes.\n")

def extract_stylometric_features(text):
    """Extract stylometric, POS, and statistical features"""
    
    try:
        doc = nlp(text[:100000])
        
        # Get sentences
        sentences = list(doc.sents)
        sentence_lengths = [len([t for t in sent if not t.is_punct and not t.is_space]) for sent in sentences]
        
        # Get all words (non-punct, non-space)
        words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
        
        if len(words) == 0:
            return {f: 0.0 for f in ['sent_len_mean', 'sent_len_std', 'sent_len_min', 'sent_len_max',
                                      'type_token_ratio', 'yules_k', 'noun_freq', 'verb_freq', 'adj_freq',
                                      'adv_freq', 'avg_word_len', 'long_word_ratio', 'short_word_ratio',
                                      'burstiness', 'entropy', 'repetition_rate', 'avg_dependency_depth']}
        
        # 1. Sentence length statistics
        sent_len_mean = np.mean(sentence_lengths) if sentence_lengths else 0
        sent_len_std = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
        sent_len_min = min(sentence_lengths) if sentence_lengths else 0
        sent_len_max = max(sentence_lengths) if sentence_lengths else 0
        
        # 2. Type-Token Ratio (vocabulary richness)
        unique_words = len(set(words))
        total_words = len(words)
        type_token_ratio = unique_words / total_words if total_words > 0 else 0
        
        # 3. Yule's K statistic (vocabulary diversity)
        word_freq = Counter(words)
        M1 = sum(word_freq.values())
        M2 = sum([freq**2 for freq in word_freq.values()])
        yules_k = 10000 * (M2 - M1) / (M1 * M1) if M1 > 0 else 0
        
        # 4. POS tag frequencies
        pos_counts = Counter([token.pos_ for token in doc if not token.is_punct and not token.is_space])
        noun_freq = pos_counts['NOUN'] / total_words if total_words > 0 else 0
        verb_freq = pos_counts['VERB'] / total_words if total_words > 0 else 0
        adj_freq = pos_counts['ADJ'] / total_words if total_words > 0 else 0
        adv_freq = pos_counts['ADV'] / total_words if total_words > 0 else 0
        
        # 5. Word length features
        word_lengths = [len(word) for word in words]
        avg_word_len = np.mean(word_lengths) if word_lengths else 0
        long_word_ratio = sum(1 for w in word_lengths if w > 6) / total_words if total_words > 0 else 0
        short_word_ratio = sum(1 for w in word_lengths if w <= 3) / total_words if total_words > 0 else 0
        
        # 6. Burstiness (variance of sentence lengths / mean)
        burstiness = sent_len_std / sent_len_mean if sent_len_mean > 0 else 0
        
        # 7. Entropy (Shannon entropy from word frequencies)
        total = sum(word_freq.values())
        entropy_val = -sum((count/total) * math.log2(count/total) for count in word_freq.values()) if total > 0 else 0
        
        # 8. Repetition rate (bigrams)
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        unique_bigrams = len(set(bigrams))
        repetition_rate = 1 - (unique_bigrams / len(bigrams)) if bigrams else 0
        
        # 9. Average dependency tree depth
        depths = []
        for sent in sentences:
            for token in sent:
                depth = 0
                current = token
                while current.head != current:
                    depth += 1
                    current = current.head
                depths.append(depth)
        avg_dependency_depth = np.mean(depths) if depths else 0
        
        return {
            'sent_len_mean': sent_len_mean,
            'sent_len_std': sent_len_std,
            'sent_len_min': sent_len_min,
            'sent_len_max': sent_len_max,
            'type_token_ratio': type_token_ratio,
            'yules_k': yules_k,
            'noun_freq': noun_freq,
            'verb_freq': verb_freq,
            'adj_freq': adj_freq,
            'adv_freq': adv_freq,
            'avg_word_len': avg_word_len,
            'long_word_ratio': long_word_ratio,
            'short_word_ratio': short_word_ratio,
            'burstiness': burstiness,
            'entropy': entropy_val,
            'repetition_rate': repetition_rate,
            'avg_dependency_depth': avg_dependency_depth
        }
    
    except Exception as e:
        print(f"Error: {e}")
        return {f: 0.0 for f in ['sent_len_mean', 'sent_len_std', 'sent_len_min', 'sent_len_max',
                                  'type_token_ratio', 'yules_k', 'noun_freq', 'verb_freq', 'adj_freq',
                                  'adv_freq', 'avg_word_len', 'long_word_ratio', 'short_word_ratio',
                                  'burstiness', 'entropy', 'repetition_rate', 'avg_dependency_depth']}

# Extract features
features_list = []

for idx in tqdm(range(len(df)), desc="Extracting stylometric features"):
    text = df.iloc[idx]['text']
    features = extract_stylometric_features(text)
    features_list.append(features)

# Convert to DataFrame
features_df = pd.DataFrame(features_list)

# Combine with existing data
result_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

# Save
result_df.to_csv('data/raw/features_all.csv', index=False)

print(f"\n✅ Stylometric feature extraction complete!")
print(f"Saved to: data/raw/features_all.csv")
print(f"Shape: {result_df.shape}")
print(f"\nNew features added ({len(features_df.columns)}):")
for col in features_df.columns:
    print(f"  - {col}")