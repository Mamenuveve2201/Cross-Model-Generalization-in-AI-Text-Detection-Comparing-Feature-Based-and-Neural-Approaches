import pandas as pd
import numpy as np
import re
import collections
import string
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("data/raw/features_all.csv")
texts = df["text"].fillna("").tolist()

STOPWORDS = set(
    "i me my myself we our ours ourselves you your yours yourself yourselves "
    "he him his himself she her hers herself it its itself they them their "
    "theirs themselves what which who whom this that these those am is are "
    "was were be been being have has had having do does did doing a an the "
    "and but if or because as until while of at by for with about against "
    "between into through during before after above below to from up down "
    "in out on off over under again further then once here there when where "
    "why how all both each few more most other some such no nor not only own "
    "same so than too very s t can will just don should now d ll m o re ve "
    "y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn "
    "shan shouldn wasn weren won wouldn".split()
)

def count_syllables(word):
    word = word.lower().strip("'")
    if not word:
        return 0
    count = len(re.findall(r"[aeiou]+", word))
    if word.endswith("e") and len(word) > 2 and count > 1:
        count -= 1
    return max(1, count)

def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def extract_new_features(text):
    if not isinstance(text, str) or not text.strip():
        return {k: 0.0 for k in [
            "hapax_ratio", "lexical_density", "syllables_per_word",
            "word_length_variance", "vocab_richness", "bigram_repetition",
            "trigram_repetition", "sentence_start_diversity",
            "punctuation_density", "question_freq", "unique_words_ratio",
            "pos_pronoun_ratio", "pos_det_ratio", "pos_conj_ratio",
            "noun_phrase_density", "named_entity_ratio"
        ]}

    words = re.findall(r"[a-zA-Z']+", text.lower())
    alpha_words = [w for w in words if re.match(r"^[a-z']+$", w)]
    n = max(len(alpha_words), 1)
    freq = collections.Counter(alpha_words)
    V = max(len(freq), 1)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip()) if s.strip()]
    n_sents = max(len(sentences), 1)

    # 1. Hapax ratio
    hapax_ratio = sum(1 for c in freq.values() if c == 1) / V

    # 2. Lexical density
    content_words = [w for w in alpha_words if w not in STOPWORDS]
    lexical_density = len(content_words) / n

    # 3. Syllables per word
    syllables_per_word = float(np.mean([count_syllables(w) for w in alpha_words])) if alpha_words else 0.0

    # 4. Word length variance
    word_lengths = [len(w) for w in alpha_words]
    word_length_variance = float(np.var(word_lengths)) if word_lengths else 0.0

    # 5. Vocabulary richness (sqrt normalization)
    vocab_richness = V / np.sqrt(n)

    # 6. Bigram repetition
    bigrams = get_ngrams(alpha_words, 2)
    bigram_repetition = (1 - len(set(bigrams)) / len(bigrams)) if bigrams else 0.0

    # 7. Trigram repetition
    trigrams = get_ngrams(alpha_words, 3)
    trigram_repetition = (1 - len(set(trigrams)) / len(trigrams)) if trigrams else 0.0

    # 8. Sentence start diversity
    first_words = [s.split()[0].lower() for s in sentences if s.split()]
    sentence_start_diversity = len(set(first_words)) / len(first_words) if first_words else 1.0

    # 9. Punctuation density
    punctuation_density = sum(1 for ch in text if ch in string.punctuation) / max(len(text), 1)

    # 10. Question frequency
    question_freq = sum(1 for s in sentences if s.strip().endswith("?")) / n_sents

    # 11. Unique words ratio
    unique_words_ratio = V / n

    # 12-14. POS ratios via spaCy
    doc = nlp(text[:3000])
    total_tokens = max(len(doc), 1)
    pos_pronoun_ratio = sum(1 for t in doc if t.pos_ == "PRON") / total_tokens
    pos_det_ratio     = sum(1 for t in doc if t.pos_ == "DET")  / total_tokens
    pos_conj_ratio    = sum(1 for t in doc if t.pos_ in {"CCONJ", "SCONJ"}) / total_tokens

    # 15. Noun phrase density
    noun_phrase_density = len(list(doc.noun_chunks)) / n_sents

    # 16. Named entity ratio
    named_entity_ratio = len(doc.ents) / total_tokens

    return {
        "hapax_ratio": hapax_ratio,
        "lexical_density": lexical_density,
        "syllables_per_word": syllables_per_word,
        "word_length_variance": word_length_variance,
        "vocab_richness": vocab_richness,
        "bigram_repetition": bigram_repetition,
        "trigram_repetition": trigram_repetition,
        "sentence_start_diversity": sentence_start_diversity,
        "punctuation_density": punctuation_density,
        "question_freq": question_freq,
        "unique_words_ratio": unique_words_ratio,
        "pos_pronoun_ratio": pos_pronoun_ratio,
        "pos_det_ratio": pos_det_ratio,
        "pos_conj_ratio": pos_conj_ratio,
        "noun_phrase_density": noun_phrase_density,
        "named_entity_ratio": named_entity_ratio,
    }

print("Extracting 16 new features from", len(texts), "samples...")
new_feats = [extract_new_features(t) for t in tqdm(texts)]
new_df = pd.DataFrame(new_feats)

result = pd.concat([df.reset_index(drop=True), new_df], axis=1)
result = result.fillna(0)
result.to_csv("data/raw/features_all.csv", index=False)

print("Done. New shape:", result.shape)
print("Features now:", result.shape[1] - 4, "(excluding metadata)")